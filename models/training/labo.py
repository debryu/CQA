import os
from loguru import logger
import torch
import random
from torch.utils.data import DataLoader, TensorDataset
from models.glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from tqdm import tqdm

from utils.lfcbm_utils import cos_similarity_cubed_single, save_activations, get_save_names, get_targets_only
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
    W_g = output_proj['path'][0]['weight']
    b_g = output_proj['path'][0]['bias']
    

    # Save resulting concepts
    with open(os.path.join(save_name, "concepts.txt"), 'w') as f:
        f.write(concepts[0])
        for concept in concepts[1:]:
            f.write('\n'+concept)
    
    torch.save(W_g, os.path.join(save_name, "W_g.pt"))
    torch.save(b_g, os.path.join(save_name, "b_g.pt"))

    return args