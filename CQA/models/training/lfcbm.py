import os
from loguru import logger
import torch
import random
from torch.utils.data import DataLoader, TensorDataset
from CQA.models.glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from tqdm import tqdm

from CQA.utils.lfcbm_utils import cos_similarity_cubed_single, save_activations, get_save_names, get_targets_only
from CQA.datasets import get_dataset
from CQA.config import folder_naming_convention, ACTIVATIONS_PATH, LABELS

def train(args):
    if not os.path.exists(args.save_dir):
        raise FileNotFoundError(f"Directory {args.save_dir} does not exist")
    if args.concept_set==None or args.concept_set=="":
        raise ValueError("Concept set cannot be empty")
    if not os.path.exists(args.concept_set):
        raise FileNotFoundError(f"Concept set {args.concept_set} does not exist")
    
    save_name = os.path.join(args.save_dir,folder_naming_convention(args))
    args.save_dir = save_name
    os.makedirs(save_name, exist_ok=True)
    # Set the activation directory inside the save directory
    if args.activation_dir == 'shared':
        args.activation_dir = ACTIVATIONS_PATH['shared']
    elif args.activation_dir == 'default':
        args.activation_dir = os.path.join(save_name, 'activations')
        os.makedirs(args.activation_dir, exist_ok=True)    
    
    similarity_fn = cos_similarity_cubed_single


    with open(args.concept_set) as f:
        concepts = f.read().split("\n")

    concepts_dict = {'raw_concepts': concepts, 'raw_dim': len(concepts)}
    classes = LABELS[args.dataset]
    logger.debug(f"Classes: {classes}")
    logger.debug(f"Concepts: {concepts_dict}")


    d_train = args.dataset + "_train"
    d_val = args.dataset + "_val"
    d_test = args.dataset + "_test"

    #save activations and get save_paths
    logger.debug(f"Saving activations for {args.clip_name} on {args.backbone} backbone")
    for d_probe in [d_train, d_val, d_test]:
        save_activations(clip_name = args.clip_name, target_name = args.backbone, 
                               target_layers = [args.feature_layer], d_probe = d_probe,
                               concept_set = args.concept_set, batch_size = args.batch_size, 
                               device = args.device, pool_mode = "avg", save_dir = args.activation_dir)
        
    target_save_name, clip_save_name, text_save_name = get_save_names(args.clip_name, args.backbone, 
                                            args.feature_layer,d_train, args.concept_set, "avg", args.activation_dir)
    val_target_save_name, val_clip_save_name, text_save_name =  get_save_names(args.clip_name, args.backbone,
                                            args.feature_layer, d_val, args.concept_set, "avg", args.activation_dir)
    test_target_save_name, test_clip_save_name, text_save_name =  get_save_names(args.clip_name, args.backbone,
                                            args.feature_layer, d_test, args.concept_set, "avg", args.activation_dir)
    
    logger.debug(f"Target save name: {target_save_name}")
    logger.debug(f"Clip save name: {clip_save_name}")
    logger.debug(f"Text save name: {text_save_name}")
    #load features
    with torch.no_grad():
        target_features = torch.load(target_save_name, map_location="cpu", weights_only=True).float()
        
        val_target_features = torch.load(val_target_save_name, map_location="cpu", weights_only=True).float()
        test_target_features = torch.load(test_target_save_name, map_location="cpu", weights_only=True).float()

        image_features = torch.load(clip_save_name, map_location="cpu", weights_only=True).float()
        image_features /= torch.norm(image_features, dim=1, keepdim=True)

        val_image_features = torch.load(val_clip_save_name, map_location="cpu", weights_only=True).float()
        val_image_features /= torch.norm(val_image_features, dim=1, keepdim=True)

        test_image_features = torch.load(test_clip_save_name, map_location="cpu", weights_only=True).float()
        test_image_features /= torch.norm(test_image_features, dim=1, keepdim=True)

        text_features = torch.load(text_save_name, map_location="cpu", weights_only=True).float()
        text_features /= torch.norm(text_features, dim=1, keepdim=True)
        
        clip_features = image_features @ text_features.T            # Namely, P
        val_clip_features = val_image_features @ text_features.T    # Namely, P_val
        #test_clip_features = test_image_features @ text_features.T

        del image_features, text_features, val_image_features, test_image_features
    
    #filter concepts not activating highly
    # For every concept (vary the samples, aka dim 0) take the top 5 values (the 5 samples that activated that concept the most)
    # Return the values and compute the mean to get the mean of the top 5 activations for each concept
    highest = torch.mean(torch.topk(clip_features, dim=0, k=5)[0], dim=0)
    logger.debug(f"Highest top 5 activations for each concept:\n{highest}")
    
    #print('Stage 0', len(concepts))
    removed_concepts_id = []
    for i, concept in enumerate(concepts):
        if highest[i]<=args.clip_cutoff:
            logger.debug("Deleting {}, CLIP top5:{:.3f}".format(concept, highest[i]))
            removed_concepts_id.append(i)
    logger.info(f'Removed concepts: {removed_concepts_id}')
    concepts = [concepts[i] for i in range(len(concepts)) if highest[i]>args.clip_cutoff]
    
    #save memory by recalculating
    del clip_features
    with torch.no_grad():
        image_features = torch.load(clip_save_name, map_location="cpu", weights_only=True).float()
        image_features /= torch.norm(image_features, dim=1, keepdim=True)

        text_features = torch.load(text_save_name, map_location="cpu", weights_only=True).float()[highest>args.clip_cutoff]
        text_features /= torch.norm(text_features, dim=1, keepdim=True)
    
        clip_features = image_features @ text_features.T                # This is the actual P, removing the concepts that are not activating highly
        del image_features, text_features
    
    val_clip_features = val_clip_features[:, highest>args.clip_cutoff]  # This is the actual P_val, removing the concepts that are not activating highly
    
    #learn projection layer
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(concepts),
                                 bias=False).to(args.device)
    opt = torch.optim.Adam(proj_layer.parameters(), lr=1e-3)
    
    indices = [ind for ind in range(len(target_features))]
    
    best_val_loss = float("inf")
    best_step = 0
    best_weights = None
    proj_batch_size = min(args.proj_batch_size, len(target_features))
    for i in tqdm(range(args.proj_steps), desc="Projection Layer Training"):
        batch = torch.LongTensor(random.sample(indices, k=proj_batch_size))
        outs = proj_layer(target_features[batch].to(args.device).detach())
        loss = -similarity_fn(clip_features[batch].to(args.device).detach(), outs)
        
        loss = torch.mean(loss)
        loss.backward()
        opt.step()
        if i%50==0 or i==args.proj_steps-1:
            with torch.no_grad():
                val_output = proj_layer(val_target_features.to(args.device).detach())
                val_loss = -similarity_fn(val_clip_features.to(args.device).detach(), val_output)
                val_loss = torch.mean(val_loss)
            if i==0:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
                logger.debug("Step:{}, Avg train similarity:{:.4f}, Avg val similarity:{:.4f}".format(best_step, -loss.cpu(),
                                                                                               -best_val_loss.cpu()))
                
            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
            else: #stop if val loss starts increasing
                break
        opt.zero_grad()
        #if i==10:       #TEMPORARY TO MAKE IT FASTER
        #    break
    proj_layer.load_state_dict({"weight":best_weights})
    logger.debug("Best step:{}, Avg val similarity:{:.4f}".format(best_step, -best_val_loss.cpu()))
    
    #delete concepts that are not interpretable
    with torch.no_grad():
        outs = proj_layer(val_target_features.to(args.device).detach())
        sim = similarity_fn(val_clip_features.to(args.device).detach(), outs)
        logger.debug(f" Similarity scores:\n{sim}")
        interpretable = sim > args.interpretability_cutoff
        
    #print('Stage 1', len(concepts))
    similarities = []
    logger.debug(f'Starting with: {len(concepts)}')
    removed_ids_second_round = []
    for i, concept in enumerate(concepts):
        similarities.append(sim[i].item())
        if sim[i]<=args.interpretability_cutoff:
            logger.debug("Deleting {}, Iterpretability:{:.3f}".format(concept, sim[i]))
            removed_ids_second_round.append(i)

    concepts = [concepts[i] for i in range(len(concepts)) if interpretable[i]]
    logger.debug(f'Ending with: {len(concepts)}')
    logger.debug(f'Removed concepts: {removed_ids_second_round} -> for interpretability')
    logger.debug(f'Size: {len(removed_ids_second_round)}')
    logger.debug(f'Similarities: {similarities}')
    
    
    del clip_features, val_clip_features
    #print(interpretable)
    #print(len(interpretable))
    W_c = proj_layer.weight[interpretable]
    logger.debug(f'Shape of the final layer {W_c.shape}')
   
    #print('W_c:', W_c.shape)
    proj_layer = torch.nn.Linear(in_features=W_c.shape[1], out_features=len(concepts), bias=False)
    proj_layer.load_state_dict({"weight":W_c})

    train_targets = get_targets_only(args.dataset, split = "train")
    val_targets = get_targets_only(args.dataset, split = "val")
    test_targets = get_targets_only(args.dataset, split = "test")
   
    with torch.no_grad():
        train_c = proj_layer(target_features.detach())
        val_c = proj_layer(val_target_features.detach())
        test_c = proj_layer(test_target_features.detach())

        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True)
        
        train_c -= train_mean
        train_c /= train_std
        
        train_y = torch.LongTensor(train_targets)

        indexed_train_ds = IndexedTensorDataset(train_c, train_y)

        val_c -= train_mean
        val_c /= train_std
        val_y = torch.LongTensor(val_targets)
        val_ds = TensorDataset(val_c,val_y)

        test_c -= train_mean
        test_c /= train_std
        test_y = torch.LongTensor(test_targets)
        print(len(test_c), len(test_y))
        test_ds = TensorDataset(test_c,test_y)

    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.saga_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.saga_batch_size, shuffle=False)

    classes = LABELS[args.dataset.split('_')[0]]
    linear = torch.nn.Linear(train_c.shape[1],len(classes)).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    metadata = {}
    metadata['max_reg'] = {}
    metadata['max_reg']['nongrouped'] = args.lam

    # Solve the GLM path
    output_proj = glm_saga(linear, indexed_train_loader, args.glm_step_size, args.n_iters, args.glm_alpha, epsilon=1, k=1,
                    val_loader=val_loader, test_loader=test_loader, do_zero=False, metadata=metadata, n_ex=train_c.shape[0], n_classes = len(classes))
                    #balancing_loss_weight = data.label_weights)
    '''
    #DEBUGGING
    train_ds = TensorDataset(train_c, train_y)
    val_ds = TensorDataset(val_c,val_y)
    test_ds = TensorDataset(test_c,test_y)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=True)
    
    #linear = torch.nn.Sequential(
    #    torch.nn.Linear(clip_features.shape[1],1000),
    #    torch.nn.ReLU(),
    #    torch.nn.Linear(1000,1000),
    #   torch.nn.ReLU(),
    #    torch.nn.Linear(1000,1000),
    #    torch.nn.ReLU(),
    #    torch.nn.Linear(1000,len(classes))
    #).to(args.device)
    from datasets import GenericDataset
    bWe = GenericDataset(args.dataset, split='train').label_occurrencies
    print(bWe)
    TbWe = torch.zeros(2).to(args.device)
    TbWe[0] = 1/bWe['0']
    TbWe[1] = 1/bWe['1']
    print(TbWe)
    #bWe = torch.ones(2).to(args.device)
    #bWe[1]=3
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean', weight=TbWe)
    from sklearn.svm import LinearSVC
    # Train LinearSVC model
    clf = LinearSVC(C= 1)
    import numpy as np
    all_x = []
    all_labels = []
    for batch in train_loader:
        x,y = batch
        x = x.to('cuda')
        y = y.to('cuda')
        # Store predictions and labels
        all_x.extend(x.cpu().numpy())  # Move to CPU & convert to NumPy
        all_labels.extend(y.cpu().numpy())  # Move to CPU & convert to NumPy
    clf.fit(all_x, all_labels)

    weights = torch.tensor(clf.coef_)
    bias = torch.tensor(clf.intercept_)
    print(weights)
    W_g = torch.cat([-weights/2,weights/2])
    print(W_g)
    b_g = torch.cat([-bias/2, bias/2])
    print(torch.topk(weights, k=3, largest=True))
    print(torch.topk(weights, k=3, largest=False))

    n_concepts_final_layer = train_c.shape[1]
    logger.debug(f"N. of concepts in the final bottleneck: {n_concepts_final_layer}")
    # add n_concepts to the args
    setattr(args, 'n_concepts_final_layer', n_concepts_final_layer)
    linear = torch.nn.Linear(train_c.shape[1],len(classes)).to(args.device)
    #linear.load_state_dict({"weight":W_g, "bias":b_g})
    

    from sklearn.metrics import classification_report
    # Initialize lists to store predictions and ground truth labels
    all_preds = []
    all_nn_preds = []
    all_labels = []
    losses = []
    for batch in test_loader:
        x,y = batch 
        #print(clf.predict(x.cpu().numpy())[0:2])
        pred_y = linear(x.to(args.device))
        #print(torch.argmax(pred_y[0:2], dim=1)[0:2])
        nn_pred_y = torch.argmax(pred_y, dim=1)
        #print(nn_pred_y)
        #loss = loss_fn(pred_y,y.to(args.device))
        #losses.append(loss.item())
        # Make predictions
        all_nn_preds.extend(nn_pred_y.cpu().numpy())
        all_preds.extend(clf.predict(x.cpu().numpy()))
        all_labels.extend(y.cpu().numpy())  # Move to CPU & convert to NumPy
    print(np.mean(losses))
    # Compute classification report
    report = classification_report(all_labels, all_preds, digits=4)
    print(report)
    #report = classification_report(all_labels, all_nn_preds, digits=4, output_dict=True)
    #print(report)
    
    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.saga_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.saga_batch_size, shuffle=False) 
    # Make linear model and zero initialize
    n_concepts_final_layer = train_c.shape[1]
    logger.debug(f"N. of concepts in the final bottleneck: {n_concepts_final_layer}")
    # add n_concepts to the args
    setattr(args, 'n_concepts_final_layer', n_concepts_final_layer)
    linear = torch.nn.Linear(train_c.shape[1],len(classes)).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    
    STEP_SIZE = 0.1
    ALPHA = 0.99
    metadata = {}
    metadata['max_reg'] = {}
    metadata['max_reg']['nongrouped'] = args.lam

    # Solve the GLM path
    output_proj = glm_saga(linear, indexed_train_loader, args.glm_step_size, args.n_iters, args.glm_alpha, epsilon=1, k=1,
                      val_loader=val_loader, test_loader=test_loader, do_zero=False, metadata=metadata, n_ex=len(target_features), n_classes = len(classes))
    '''
    W_g = output_proj['path'][0]['weight']
    b_g = output_proj['path'][0]['bias']
    

    # Save resulting concepts
    with open(os.path.join(save_name, "concepts.txt"), 'w') as f:
        f.write(concepts[0])
        for concept in concepts[1:]:
            f.write('\n'+concept)
    
    torch.save(train_mean, os.path.join(save_name, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_name, "proj_std.pt"))
    torch.save(W_c, os.path.join(save_name ,"W_c.pt"))
    torch.save(W_g, os.path.join(save_name, "W_g.pt"))
    torch.save(b_g, os.path.join(save_name, "b_g.pt"))

    return args

def train_last_layer(args):
    ''' #########################################
        ####        TRAIN LAST LAYER         ####
        #########################################
    '''
    pass