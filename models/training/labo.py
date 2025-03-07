import os

import torch.functional
import torch.nn.functional as F
import torch.optim
from loguru import logger
import torch
import random
from torch.utils.data import DataLoader, TensorDataset
from models.glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from tqdm import tqdm
import copy
from utils.lfcbm_utils import cos_similarity_cubed_single, save_activations, get_save_names, get_targets_only
from datasets import GenericDataset
from datasets import get_dataset
from config import folder_naming_convention, ACTIVATIONS_PATH, LABELS

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

    with open(args.concept_set) as f:
        concepts = f.read().split("\n")

    concepts_dict = {'raw_concepts': concepts, 'raw_dim': len(concepts)}
    classes = LABELS[args.dataset.split("_")[0]]
    logger.debug(f"Classes: {classes}")
    logger.debug(f"Concepts: {concepts_dict}")


    d_train = args.dataset + "_train"
    d_val = args.dataset + "_val"
    d_test = args.dataset + "_test"

    #save activations and get save_paths
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
        test_clip_features = test_image_features @ text_features.T

        del image_features, text_features, val_image_features

    train_targets = get_targets_only(args.dataset, "train")
    val_targets = get_targets_only(args.dataset, "val")
    test_targets = get_targets_only(args.dataset, "test")
    
    from utils.labo_utils import FinalLayer
    import numpy as np
    train_ds_labo = TensorDataset(clip_features, torch.tensor(train_targets))
    val_ds_labo = TensorDataset(val_clip_features, torch.tensor(val_targets))
    test_ds_labo = TensorDataset(test_clip_features, torch.tensor(test_targets))
    train_loader_labo = DataLoader(train_ds_labo, 64, shuffle=True)
    val_loader_labo = DataLoader(val_ds_labo, 64, shuffle=False)
    test_loader_labo = DataLoader(test_ds_labo, 64, shuffle=False)
    final_layer = FinalLayer(clip_features.shape[1],len(classes))
    optim = torch.optim.Adam(final_layer.parameters(), lr=0.001)
    best_loss = 100000
    patience = 0
    for e in range(1000):
        t_losses = []
        for batch in train_loader_labo:
            clip, labl = batch
            sim = final_layer(clip*100)
            loss = torch.nn.functional.cross_entropy(sim, labl)
            optim.zero_grad()
            loss.backward()
            optim.step()
            t_losses.append(loss.item())
        v_losses = []
        for batch in val_loader_labo:
            clip, labl = batch
            sim = final_layer(clip*100)
            loss = torch.nn.functional.cross_entropy(sim, labl)
            v_losses.append(loss.item())
        logger.info(f"Train loss: {np.mean(t_losses)} Val loss: {np.mean(v_losses)}. Patience: {patience}")
        patience += 1
        if np.mean(v_losses) < best_loss:
            best_loss = np.mean(v_losses)
            W_g = final_layer.asso_mat
            patience = 0
        if patience > 6:
            break
    b_g = torch.zeros(len(classes))
    predictions = []
    labels = []
    for batch in test_loader_labo:
        clip, labl = batch
        #print(clip.shape)
        #print(labl)
        preds = final_layer(clip*100)
        #print(preds)
        #print(preds.shape)
        preds = torch.argmax(preds.cpu(), dim=1)
        #print(preds)
        #print(preds)
        predictions.extend(preds)
        labels.extend(labl.cpu().tolist())
    #input("...")
    from sklearn.metrics import classification_report
    print(classification_report(labels,predictions))
    '''
    #
    with torch.no_grad():
        train_y = torch.LongTensor(train_targets)
        indexed_train_ds = IndexedTensorDataset(clip_features, train_y)
        val_y = torch.LongTensor(val_targets)
        val_ds = TensorDataset(val_clip_features,val_y)
        test_y = torch.LongTensor(test_targets)
        test_ds = TensorDataset(test_clip_features,test_y)

    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.saga_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds,batch_size=args.saga_batch_size, shuffle=False)

    # Make linear model and zero initialize
    n_concepts_final_layer = clip_features.shape[1]
    logger.debug(f"N. of concepts in the final bottleneck: {n_concepts_final_layer}")
    # add n_concepts to the args
    setattr(args, 'n_concepts_final_layer', n_concepts_final_layer)
    linear = torch.nn.Linear(clip_features.shape[1],len(classes)).to(args.device)
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
    #
    
    print(clip_features.shape)
    print(val_clip_features.shape)
    print(test_clip_features.shape)
    #DEBUGGING
    train_y = torch.LongTensor(train_targets)
    train_ds = TensorDataset(clip_features, train_y)
    val_y = torch.LongTensor(val_targets)
    val_ds = TensorDataset(val_clip_features,val_y)
    test_y = torch.LongTensor(test_targets)
    test_ds = TensorDataset(test_clip_features,test_y)
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
    TbWe[0] = 48000/bWe['0']
    TbWe[1] = 48000/bWe['1']
    print(TbWe)
    #bWe = torch.ones(2).to(args.device)
    #bWe[1]=3
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean', weight=TbWe)
    from sklearn.svm import LinearSVC
    # Train LinearSVC model
    clf = LinearSVC(C = 1)
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

    n_concepts_final_layer = clip_features.shape[1]
    logger.debug(f"N. of concepts in the final bottleneck: {n_concepts_final_layer}")
    # add n_concepts to the args
    setattr(args, 'n_concepts_final_layer', n_concepts_final_layer)
    linear = torch.nn.Linear(clip_features.shape[1],len(classes)).to(args.device)
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
    '''
    # Save resulting concepts
    with open(os.path.join(save_name, "concepts.txt"), 'w') as f:
        f.write(concepts[0])
        for concept in concepts[1:]:
            f.write('\n'+concept)
    
    print(W_g)
    torch.save(W_g, os.path.join(save_name, "W_g.pt"))
    torch.save(b_g, os.path.join(save_name, "b_g.pt"))

    return args

def train_last_layer(args):
    ''' #########################################
        ####        TRAIN LAST LAYER         ####
        #########################################
    '''
    pass