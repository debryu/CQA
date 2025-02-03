import os
import torch
from torchvision import datasets, transforms, models
from pytorchcv.model_provider import get_model as ptcv_get_model
import utils.clip as clip
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import get_dataset
from loguru import logger
from functools import partial
from utils.utils import get_resnet_imagenet_preprocess

PM_SUFFIX = {"max":"_max", "avg":""}

'''
Avoid anonymous functions in the code. They can't be pickled and will cause issues when saving the model.
These two functions are created to avoid the issue.
'''
def target_model_function(model, x):
    return model.encode_image(x).float()
'''---------------------------------'''

def get_targets_only(dataset_name,split):
    logger.debug(f'Retrieving labels of {dataset_name} {split}...')
    pil_data = get_dataset(ds_name=dataset_name,split=split)
    targets = []
    for i in tqdm(range(len(pil_data)), desc='Retrieving labels'):
        _, _, label = pil_data[i]
        targets.append(label)
        #if i==10:
        #    break
    return targets

def get_target_model(target_name, device):
    if target_name.startswith("clip_"):
        target_name = target_name[5:]
        model, preprocess = clip.load(target_name, device=device)
        target_model = partial(target_model_function, model)
        #target_model = lambda x: model.encode_image(x).float()
    
    elif target_name == 'resnet18_places': 
        target_model = models.resnet18(pretrained=False, num_classes=365).to(device)
        state_dict = torch.load('data/resnet18_places365.pth.tar')['state_dict']
        new_state_dict = {}
        for key in state_dict:
            if key.startswith('module.'):
                new_state_dict[key[7:]] = state_dict[key]
        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
        
    elif target_name == 'resnet18_cub':
        target_model = ptcv_get_model("resnet18_cub", pretrained=True).to(device)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
    
    elif target_name.endswith("_v2"):
        target_name = target_name[:-3]
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V2".format(target_name_cap))
        target_model = eval("models.{}(weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()
        
    else:
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()
    
    return target_model, preprocess


def cos_similarity_cubed_single(clip_feats, target_feats):
    """
    Substract mean from each vector, then raises to third power and compares cos similarity
    Does not modify any tensors in place
    Only compares first neuron to first concept etc.
    """

    clip_feats = clip_feats.float()
    clip_feats = clip_feats - torch.mean(clip_feats, dim=0, keepdim=True)
    target_feats = target_feats - torch.mean(target_feats, dim=0, keepdim=True)

    clip_feats = clip_feats**3
    target_feats = target_feats**3

    clip_feats = clip_feats/torch.norm(clip_feats, p=2, dim=0, keepdim=True)
    target_feats = target_feats/torch.norm(target_feats, p=2, dim=0, keepdim=True)

    similarities = torch.sum(target_feats*clip_feats, dim=0)
    return similarities

def cos_similarity_cubed(clip_feats, target_feats, device='cuda', batch_size=10000, min_norm=1e-3):
    """
    Substract mean from each vector, then raises to third power and compares cos similarity
    Does not modify any tensors in place
    """
    with torch.no_grad():
        torch.cuda.empty_cache()
        
        clip_feats = clip_feats - torch.mean(clip_feats, dim=0, keepdim=True)
        target_feats = target_feats - torch.mean(target_feats, dim=0, keepdim=True)
        
        clip_feats = clip_feats**3
        target_feats = target_feats**3
        
        clip_feats = clip_feats/torch.clip(torch.norm(clip_feats, p=2, dim=0, keepdim=True), min_norm)
        target_feats = target_feats/torch.clip(torch.norm(target_feats, p=2, dim=0, keepdim=True), min_norm)
        
        similarities = []
        for t_i in tqdm(range(math.ceil(target_feats.shape[1]/batch_size))):
            curr_similarities = []
            curr_target = target_feats[:, t_i*batch_size:(t_i+1)*batch_size].to(device).T
            for c_i in range(math.ceil(clip_feats.shape[1]/batch_size)):
                curr_similarities.append(curr_target @ clip_feats[:, c_i*batch_size:(c_i+1)*batch_size].to(device))
            similarities.append(torch.cat(curr_similarities, dim=1))
    return torch.cat(similarities, dim=0)

def cos_similarity(clip_feats, target_feats, device='cuda'):
    with torch.no_grad():
        clip_feats = clip_feats / torch.norm(clip_feats, p=2, dim=0, keepdim=True)
        target_feats = target_feats / torch.norm(target_feats, p=2, dim=0, keepdim=True)
        
        batch_size = 10000
        
        similarities = []
        for t_i in tqdm(range(math.ceil(target_feats.shape[1]/batch_size))):
            curr_similarities = []
            curr_target = target_feats[:, t_i*batch_size:(t_i+1)*batch_size].to(device).T
            for c_i in range(math.ceil(clip_feats.shape[1]/batch_size)):
                curr_similarities.append(curr_target @ clip_feats[:, c_i*batch_size:(c_i+1)*batch_size].to(device))
            similarities.append(torch.cat(curr_similarities, dim=1))
    return torch.cat(similarities, dim=0)

def soft_wpmi(clip_feats, target_feats, top_k=100, a=10, lam=1, device='cuda',
                        min_prob=1e-7, p_start=0.998, p_end=0.97):
    
    with torch.no_grad():
        torch.cuda.empty_cache()
        clip_feats = torch.nn.functional.softmax(a*clip_feats, dim=1)

        inds = torch.topk(target_feats, dim=0, k=top_k)[1]
        prob_d_given_e = []

        p_in_examples = p_start-(torch.arange(start=0, end=top_k)/top_k*(p_start-p_end)).unsqueeze(1).to(device)
        for orig_id in tqdm(range(target_feats.shape[1])):
            
            curr_clip_feats = clip_feats.gather(0, inds[:,orig_id:orig_id+1].expand(-1,clip_feats.shape[1])).to(device)
            
            curr_p_d_given_e = 1+p_in_examples*(curr_clip_feats-1)
            curr_p_d_given_e = torch.sum(torch.log(curr_p_d_given_e+min_prob), dim=0, keepdim=True)
            prob_d_given_e.append(curr_p_d_given_e)
            torch.cuda.empty_cache()

        prob_d_given_e = torch.cat(prob_d_given_e, dim=0)
        print(prob_d_given_e.shape)
        #logsumexp trick to avoid underflow
        prob_d = (torch.logsumexp(prob_d_given_e, dim=0, keepdim=True) - 
                  torch.log(prob_d_given_e.shape[0]*torch.ones([1]).to(device)))
        mutual_info = prob_d_given_e - lam*prob_d
    return mutual_info

def wpmi(clip_feats, target_feats, top_k=28, a=2, lam=0.6, device='cuda', min_prob=1e-7):
    
    with torch.no_grad():
        torch.cuda.empty_cache()
        
        clip_feats = torch.nn.functional.softmax(a*clip_feats, dim=1)

        inds = torch.topk(target_feats, dim=0, k=top_k)[1]
        prob_d_given_e = []

        for orig_id in tqdm(range(target_feats.shape[1])):
            torch.cuda.empty_cache()
            curr_clip_feats = clip_feats.gather(0, inds[:,orig_id:orig_id+1].expand(-1,clip_feats.shape[1])).to(device)
            curr_p_d_given_e = torch.sum(torch.log(curr_clip_feats+min_prob), dim=0, keepdim=True)
            prob_d_given_e.append(curr_p_d_given_e)

        prob_d_given_e = torch.cat(prob_d_given_e, dim=0)
        #logsumexp trick to avoid underflow
        prob_d = (torch.logsumexp(prob_d_given_e, dim=0, keepdim=True) -
                  torch.log(prob_d_given_e.shape[0]*torch.ones([1]).to(device)))

        mutual_info = prob_d_given_e - lam*prob_d
    return mutual_info

def rank_reorder(clip_feats, target_feats, device="cuda", p=3, top_fraction=0.05, scale_p=0.5):
    """
    top fraction: percentage of mostly highly activating target images to use for eval. Between 0 and 1
    """
    with torch.no_grad():
        batch = 1500
        errors = []
        top_n = int(target_feats.shape[0]*top_fraction)
        target_feats, inds = torch.topk(target_feats, dim=0, k=top_n)

        for orig_id in tqdm(range(target_feats.shape[1])):
            clip_indices = clip_feats.gather(0, inds[:, orig_id:orig_id+1].expand([-1,clip_feats.shape[1]])).to(device)
            #calculate the average probability score of the top neurons for each caption
            avg_clip = torch.mean(clip_indices, dim=0, keepdim=True)
            clip_indices = torch.argsort(clip_indices, dim=0)
            clip_indices = torch.argsort(clip_indices, dim=0)
            curr_errors = []
            target = target_feats[:, orig_id:orig_id+1].to(device)
            sorted_target = torch.flip(target, dims=[0])

            baseline_diff = sorted_target - torch.cat([sorted_target[torch.randperm(len(sorted_target))] for _ in range(5)], dim=1)
            baseline_diff = torch.mean(torch.abs(baseline_diff)**p)
            torch.cuda.empty_cache()

            for i in range(math.ceil(clip_indices.shape[1]/batch)):

                clip_id = (clip_indices[:, i*batch:(i+1)*batch])
                reorg = sorted_target.expand(-1, batch).gather(dim=0, index=clip_id)
                diff = (target-reorg)
                curr_errors.append(torch.mean(torch.abs(diff)**p, dim=0, keepdim=True)/baseline_diff)
            errors.append(torch.cat(curr_errors, dim=1)/(avg_clip)**scale_p)

        errors = torch.cat(errors, dim=0)
    return -errors

def save_target_activations(target_model, dataset, save_name, target_layers = ["layer4"], batch_size = 1000,
                            device = "cuda", pool_mode='avg'):
    """
    save_name: save_file path, should include {} which will be formatted by layer names
    """
    _make_save_dir(save_name)
    save_names = {}    
    for target_layer in target_layers:
        save_names[target_layer] = save_name.format(target_layer)
        
    if _all_saved(save_names):
        return
    
    all_features = {target_layer:[] for target_layer in target_layers}
    
    hooks = {}
    for target_layer in target_layers:
        command = "target_model.{}.register_forward_hook(get_activation(all_features[target_layer], pool_mode))".format(target_layer)
        hooks[target_layer] = eval(command)
    
    with torch.no_grad():
        for images, _, _ in tqdm(DataLoader(dataset, batch_size)):
            features = target_model(images.to(device))
    
    for target_layer in target_layers:
        torch.save(torch.cat(all_features[target_layer]), save_names[target_layer])
        hooks[target_layer].remove()
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return

def save_clip_image_features(model, dataset, save_name, batch_size=1000 , device = "cuda"):
    logger.debug(f"Saving image features to {save_name}")
    _make_save_dir(save_name)
    all_features = []
    
    if os.path.exists(save_name):
        return
    
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        for images, _, _ in tqdm(DataLoader(dataset, batch_size)):        # Removed num_workers=8, pin_memory=True
            features = model.encode_image(images.to(device))
            all_features.append(features.cpu())

    torch.save(torch.cat(all_features), save_name)
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return

def save_clip_text_features(model, text, save_name, batch_size=1000):
    logger.debug(f"Saving text features to {save_name}")
    if os.path.exists(save_name):
        return
    _make_save_dir(save_name)
    text_features = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(text)/batch_size))):
            text_features.append(model.encode_text(text[batch_size*i:batch_size*(i+1)]))
    text_features = torch.cat(text_features, dim=0)
    torch.save(text_features, save_name)
    del text_features
    torch.cuda.empty_cache()
    return

def save_activations(clip_name, target_name, target_layers, d_probe, 
                     concept_set, batch_size, device, pool_mode, save_dir):
    
    
    target_save_name, clip_save_name, text_save_name = get_save_names(clip_name, target_name, 
                                                                    "{}", d_probe, concept_set, 
                                                                      pool_mode, save_dir)
    save_names = {"clip": clip_save_name, "text": text_save_name}
    for target_layer in target_layers:
        save_names[target_layer] = target_save_name.format(target_layer)
        
    if _all_saved(save_names):
        logger.debug(f"Activations already saved for {d_probe}")
        return
    
    clip_model, clip_preprocess = clip.load(clip_name, device=device)
    
    if target_name.startswith("clip_"):
        target_model, target_preprocess = clip.load(target_name[5:], device=device)
    else:
        target_model, target_preprocess = get_target_model(target_name, device)
    
    
    ds_name = d_probe.split("_")[0]
    split = d_probe.split("_")[-1]
    logger.debug(f"Saving activations for {ds_name} {split}...")

    #setup data
    data_c = get_dataset(ds_name=ds_name, split=split, transform=clip_preprocess)
    data_t = get_dataset(ds_name=ds_name, split=split, transform=target_preprocess)
    
    with open(concept_set, 'r') as f: 
        words = (f.read()).split('\n')
    text = clip.tokenize(["{}".format(word) for word in words]).to(device)
    
    save_clip_text_features(clip_model, text, text_save_name, batch_size)
    
    save_clip_image_features(clip_model, data_c, clip_save_name, batch_size, device)
    if target_name.startswith("clip_"):
        save_clip_image_features(target_model, data_t, target_save_name, batch_size, device)
    else:
        save_target_activations(target_model, data_t, target_save_name, target_layers,
                                batch_size, device, pool_mode)
    
    return
    
    
def get_similarity_from_activations(target_save_name, clip_save_name, text_save_name, similarity_fn, 
                                   return_target_feats=True):
    image_features = torch.load(clip_save_name)
    text_features = torch.load(text_save_name)
    with torch.no_grad():
        image_features /= image_features.norm(dim=-1, keepdim=True).float()
        text_features /= text_features.norm(dim=-1, keepdim=True).float()
        clip_feats = (image_features @ text_features.T)
    del image_features, text_features
    torch.cuda.empty_cache()
    
    target_feats = torch.load(target_save_name)
    similarity = similarity_fn(clip_feats, target_feats)
    
    del clip_feats
    torch.cuda.empty_cache()
    
    if return_target_feats:
        return similarity, target_feats
    else:
        del target_feats
        torch.cuda.empty_cache()
        return similarity
    
def get_activation(outputs, mode):
    '''
    mode: how to pool activations: one of avg, max
    for fc neurons does no pooling
    '''
    if mode=='avg':
        def hook(model, input, output):
            if len(output.shape)==4:
                outputs.append(output.mean(dim=[2,3]).detach().cpu())
            elif len(output.shape)==2:
                outputs.append(output.detach().cpu())
    elif mode=='max':
        def hook(model, input, output):
            if len(output.shape)==4:
                outputs.append(output.amax(dim=[2,3]).detach().cpu())
            elif len(output.shape)==2:
                outputs.append(output.detach().cpu())
    return hook

    
def get_save_names(clip_name, target_name, target_layer, d_probe, concept_set, pool_mode, save_dir):
    
    if target_name.startswith("clip_"):
        target_save_name = "{}/{}_{}.pt".format(save_dir, d_probe, target_name.replace('/', ''))
    else:
        target_save_name = "{}/{}_{}_{}{}.pt".format(save_dir, d_probe, target_name, target_layer,
                                                 PM_SUFFIX[pool_mode])
    clip_save_name = "{}/{}_clip_{}.pt".format(save_dir, d_probe, clip_name.replace('/', ''))
    concept_set_name = (concept_set.split("/")[-1]).split(".")[0]
    text_save_name = "{}/{}_conceptset_{}.pt".format(save_dir, concept_set_name, clip_name.replace('/', ''))
    
    return target_save_name, clip_save_name, text_save_name

    
def _all_saved(save_names):
    """
    save_names: {layer_name:save_path} dict
    Returns True if there is a file corresponding to each one of the values in save_names,
    else Returns False
    """
    for save_name in save_names.values():
        if not os.path.exists(save_name):
            return False
    return True

def _make_save_dir(save_name):
    """
    creates save directory if one does not exist
    save_name: full save path
    """
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return

def get_accuracy_cbm(model, dataset, device, batch_size=250, num_workers=2):
    correct = 0
    total = 0
    for images, labels, _ in tqdm(DataLoader(dataset, batch_size)):#EDIT, num_workers=num_workers,pin_memory=True)):
        with torch.no_grad():
            #outs = target_model(images.to(device))
            outs, _ = model(images.to(device))
            pred = torch.argmax(outs, dim=1)
            correct += torch.sum(pred.cpu()==labels)
            total += len(labels)
    return correct/total

def get_preds_cbm(model, dataset, device, batch_size=250, num_workers=2):
    preds = []
    for images, labels, _ in tqdm(DataLoader(dataset, batch_size)): #EDIT, num_workers=num_workers,pin_memory=True)):
        with torch.no_grad():
            outs, _ = model(images.to(device))
            pred = torch.argmax(outs, dim=1)
            preds.append(pred.cpu())
    preds = torch.cat(preds, dim=0)
    return preds

def get_concept_act_by_pred(model, dataset, device):
    preds = []
    concept_acts = []
    for images, labels, _ in tqdm(DataLoader(dataset, 500)):#EDIT, num_workers=8, pin_memory=True)):
        with torch.no_grad():
            outs, concept_act = model(images.to(device))
            concept_acts.append(concept_act.cpu())
            pred = torch.argmax(outs, dim=1)
            preds.append(pred.cpu())
    preds = torch.cat(preds, dim=0)
    concept_acts = torch.cat(concept_acts, dim=0)
    concept_acts_by_pred=[]
    for i in range(torch.max(pred)+1):
        concept_acts_by_pred.append(torch.mean(concept_acts[preds==i], dim=0))
    concept_acts_by_pred = torch.stack(concept_acts_by_pred, dim=0)
    return concept_acts_by_pred