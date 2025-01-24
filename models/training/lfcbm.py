import os
from loguru import logger
import torch
import random
from torch.utils.data import DataLoader, TensorDataset
from models.glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from tqdm import tqdm

from utils.lfcbm_utils import cos_similarity_cubed_single, save_activations, get_save_names, get_targets_only
from datasets import get_dataset
from config import folder_naming_convention, ACTIVATIONS_PATH

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
    classes = get_dataset(args.dataset).classes
    logger.debug(f"Classes: {classes}")
    logger.debug(f"Concepts: {concepts_dict}")


    d_train = args.dataset + "_train"
    d_val = args.dataset + "_val"

    #save activations and get save_paths
    logger.debug(f"Saving activations for {args.clip_name} on {args.backbone} backbone")
    for d_probe in [d_train, d_val]:
        save_activations(clip_name = args.clip_name, target_name = args.backbone, 
                               target_layers = [args.feature_layer], d_probe = d_probe,
                               concept_set = args.concept_set, batch_size = args.batch_size, 
                               device = args.device, pool_mode = "avg", save_dir = args.activation_dir)
        
    target_save_name, clip_save_name, text_save_name = get_save_names(args.clip_name, args.backbone, 
                                            args.feature_layer,d_train, args.concept_set, "avg", args.activation_dir)
    val_target_save_name, val_clip_save_name, text_save_name =  get_save_names(args.clip_name, args.backbone,
                                            args.feature_layer, d_val, args.concept_set, "avg", args.activation_dir)
    
    logger.debug(f"Target save name: {target_save_name}")
    logger.debug(f"Clip save name: {clip_save_name}")
    logger.debug(f"Text save name: {text_save_name}")
    #load features
    with torch.no_grad():
        target_features = torch.load(target_save_name, map_location="cpu", weights_only=True).float()
        
        val_target_features = torch.load(val_target_save_name, map_location="cpu", weights_only=True).float()

        image_features = torch.load(clip_save_name, map_location="cpu", weights_only=True).float()
        image_features /= torch.norm(image_features, dim=1, keepdim=True)

        val_image_features = torch.load(val_clip_save_name, map_location="cpu", weights_only=True).float()
        val_image_features /= torch.norm(val_image_features, dim=1, keepdim=True)

        text_features = torch.load(text_save_name, map_location="cpu", weights_only=True).float()
        text_features /= torch.norm(text_features, dim=1, keepdim=True)
        
        clip_features = image_features @ text_features.T            # Namely, P
        val_clip_features = val_image_features @ text_features.T    # Namely, P_val

        del image_features, text_features, val_image_features
    
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
    logger.info("Best step:{}, Avg val similarity:{:.4f}".format(best_step, -best_val_loss.cpu()))
    
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

    train_targets = get_targets_only(args.dataset, "train")
    
    val_targets = get_targets_only(args.dataset, "val")
   
    with torch.no_grad():
        train_c = proj_layer(target_features.detach())
        val_c = proj_layer(val_target_features.detach())

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

    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.saga_batch_size, shuffle=False)
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
    output_proj = glm_saga(linear, indexed_train_loader, STEP_SIZE, args.n_iters, ALPHA, epsilon=1, k=1,
                      val_loader=val_loader, do_zero=False, metadata=metadata, n_ex=len(target_features), n_classes = len(classes))
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