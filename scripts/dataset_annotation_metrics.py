
from config import CONCEPT_SETS, LLM_GENERATED_ANNOTATIONS
from loguru import logger
import scripts.utils
import torch
from datasets import GenericDataset
from sklearn.metrics import classification_report
from utils.eval_models import train_LR_global, train_LR_on_concepts
import numpy as np
import os
import copy
dataset = 'cub'
device = 'cuda'
confidence_threshold = 0.15
annotation_dir = "./data/VLG_annotations/"
activation_dir = "./data/activations/"
concept_set = CONCEPT_SETS[dataset]


def compute(loader, dataset_name, n_concepts, name = ''):
    #print(dataset_name)
    gt_data = GenericDataset(dataset_name, split = 'train')
    index = 0
    all_preds = []
    all_gts = []
    for sample in loader:
        _,gt_concepts,gt_labels = gt_data[index]
        _,pred_concepts,pred_labels = sample
        pred_concepts = pred_concepts.squeeze().cpu()
        gt_concepts = gt_concepts.cpu()
        #print(sample)
        all_preds.append(pred_concepts)
        all_gts.append(gt_concepts)
        index += 1
        if index > 100:
            break
    all_preds = torch.stack(all_preds, dim=0)
    all_gts = torch.stack(all_gts, dim=0)
    if dataset == 'shapes3d':
        pass
    else:
        macro_avg_precision = []
        macro_avg_recall = []
        micro_avg_precision = []
        micro_avg_recall = []
        avg_f1 = []
        for i in range(n_concepts):
            report = classification_report(all_gts[:,i],all_preds[:,i], output_dict=True)
            #print(classification_report(all_gts[:,i],all_preds[:,i]))
            macro_avg_precision.append(report['macro avg']['precision'])
            macro_avg_recall.append(report['macro avg']['recall'])
            micro_avg_precision.append(report['weighted avg']['precision'])
            micro_avg_recall.append(report['weighted avg']['recall'])
            avg_f1.append(report['macro avg']['f1-score'])
            #print(report)
        print(f"Annotator {name}:")
        print("Macro average Precision:",np.mean(macro_avg_precision), "  Max:", np.max(macro_avg_precision), "  Min:", np.min(macro_avg_precision))
        print("Macro average Recall:",np.mean(macro_avg_recall), "  Max:", np.max(macro_avg_recall), "  Min:", np.min(macro_avg_recall))
        print("Micro average Precision:",np.mean(micro_avg_precision), "  Max:", np.max(micro_avg_precision), "  Min:", np.min(micro_avg_precision))
        print("Micro average Recall:",np.mean(micro_avg_recall), "  Max:", np.max(micro_avg_recall), "  Min:", np.min(micro_avg_recall))
        print(f"F1 {np.mean(avg_f1)}")
        print("\n")

def GDino():
    raw_concepts = scripts.utils.get_concepts(concept_set)
    n_concepts = len(raw_concepts)
    # It shouldn't matter what backbone is being used
    backbone = scripts.utils.BackboneCLIP('clip_RN50', use_penultimate=False, device=device)
    logger.debug(f"Raw concepts n.{len(raw_concepts)}")
    (
        concepts,
        concept_counts,
        filtered_concepts,
    ) = scripts.utils.get_filtered_concepts_and_counts(
        dataset,
        raw_concepts,
        preprocess=backbone.preprocess,
        val_split=0.0,
        batch_size=64,
        num_workers=1,
        confidence_threshold=confidence_threshold,
        label_dir=annotation_dir,
        use_allones=False,
        seed=42,
        remove_never_seen=False
    )

    augmented_train_cbl_loader = scripts.utils.get_concept_dataloader(
            dataset,
            "train",
            concepts,
            preprocess=backbone.preprocess,
            val_split=0.0,
            batch_size=1,
            num_workers=1,
            shuffle=True,  # shuffle for training
            confidence_threshold=confidence_threshold,
            crop_to_concept_prob=0.0,  # crop to concept
            label_dir=annotation_dir,
            use_allones=False,
            seed=42,
        )

    compute(augmented_train_cbl_loader, dataset, n_concepts, name = 'GDINO')





def LLava():
    if dataset=='cub':
        used_indexes = [0,4796]
    elif dataset == 'celeba':
        pass
    else:
        pass
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    gt_data = GenericDataset(ds_name=f"{dataset}", split='train', transform = transform)
    concepts = torch.load(os.path.join(LLM_GENERATED_ANNOTATIONS,f"{dataset}_train_{used_indexes[0]}_{used_indexes[1]}.pth"), weights_only=True) 
    args = {"original_ds":gt_data, "train_concepts":concepts}
    oracle_data = scripts.utils.AnnotatedDataset(**args)
    
    loader = torch.utils.data.DataLoader(oracle_data, batch_size=1, shuffle=False)
    n_concepts = oracle_data[0][1].shape[0]
    compute(loader, dataset, n_concepts, name = 'LLava')
    pass    


def Clip():
    d_train = dataset + "_train"
    clip_name = "ViT-B/16"
    if dataset == 'cub':
        c_set = os.path.join(concept_set,'cub_preprocess.txt')
        f_layer = 'features.final_pool'
        backb = 'resnet18_cub'
    if dataset == 'celeba':
        c_set = os.path.join(concept_set,'handmade.txt')
        f_layer = 'layer4'
        backb = 'clip_RN50'
        
    target_save_name, clip_save_name, text_save_name = scripts.utils.get_save_names(clip_name, backb, 
                                            f_layer,d_train, c_set, "avg", activation_dir)
    
    logger.debug(f"Target save name: {target_save_name}")
    logger.debug(f"Clip save name: {clip_save_name}")
    logger.debug(f"Text save name: {text_save_name}")

    #load features
    with torch.no_grad():
        target_features = torch.load(target_save_name, map_location="cpu", weights_only=True).float()

        image_features = torch.load(clip_save_name, map_location="cpu", weights_only=True).float()
        image_features /= torch.norm(image_features, dim=1, keepdim=True)


        text_features = torch.load(text_save_name, map_location="cpu", weights_only=True).float()
        text_features /= torch.norm(text_features, dim=1, keepdim=True)
        
        clip_features = image_features @ text_features.T    

        del image_features

    #----------------- Without Training
    cf = copy.deepcopy(clip_features)
    probs = torch.nn.functional.sigmoid(cf)   
    preds = (probs > 0.5).long()
    clip_ds = scripts.utils.ClipDataset(preds)
    loader = torch.utils.data.DataLoader(clip_ds,batch_size=1,shuffle=False)
    original_ds = GenericDataset(ds_name=dataset, split = 'train')
    n_concepts = original_ds[0][1].shape[0]
    compute(loader, dataset, n_concepts=n_concepts, name='CLIP - raw')

    #----------------- Training one param
    cf = copy.deepcopy(clip_features)
    targets = []
    for i in range(len(original_ds)):
        _,concepts,_ = original_ds[i]
        targets.append(concepts)
    targets = torch.stack(targets, dim=0)
    W,B = train_LR_global(cf,targets)
    cf *= W
    cf += B
    #
    probs = torch.nn.functional.sigmoid(cf)   
    preds = (probs > 0.5).long()
    clip_ds = scripts.utils.ClipDataset(preds)
    loader = torch.utils.data.DataLoader(clip_ds,batch_size=1,shuffle=False)
    original_ds = GenericDataset(ds_name=dataset, split = 'train')
    n_concepts = original_ds[0][1].shape[0]
    compute(loader, dataset, n_concepts=n_concepts, name='CLIP - one LR')

    #----------------- Training one param for each concept
    cf = copy.deepcopy(clip_features)
    targets = []
    for i in range(len(original_ds)):
        _,concepts,_ = original_ds[i]
        targets.append(concepts)
    targets = torch.stack(targets, dim=0)
    logger.info("Training Logistic Regression on All Concepts")
    W,B = train_LR_on_concepts(cf, targets)
    cf *= W
    cf += B
    #
    probs = torch.nn.functional.sigmoid(cf)   
    preds = (probs > 0.5).long()
    clip_ds = scripts.utils.ClipDataset(preds)
    loader = torch.utils.data.DataLoader(clip_ds,batch_size=1,shuffle=False)
    original_ds = GenericDataset(ds_name=dataset, split = 'train')
    n_concepts = original_ds[0][1].shape[0]
    compute(loader, dataset, n_concepts=n_concepts, name='CLIP - LR on all')




if __name__ == '__main__':
    #GDino()
    #LLava()
    Clip()