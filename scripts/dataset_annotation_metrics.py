
from config import CONCEPT_SETS, LLM_GENERATED_ANNOTATIONS, ACTIVATIONS_PATH, DINO_GENERATED_ANNOTATIONS
from loguru import logger
from torchvision import transforms
import scripts.utils
import torchvision
import torch
from datasets import GenericDataset
from sklearn.metrics import classification_report
from utils.eval_models import train_LR_global, train_LR_on_concepts, train_LR_on_concepts_shapes3d
import numpy as np
import os
import copy
from tqdm import tqdm
from matplotlib import pyplot as plt
dataset = 'cub'
device = 'cuda'
confidence_threshold = 0.10
annotation_dir = DINO_GENERATED_ANNOTATIONS

activation_dir = ACTIVATIONS_PATH['shared']
concept_set = CONCEPT_SETS[dataset]


def compute(loader, dataset_name, n_concepts, name = ''):
    #print(dataset_name)
    if dataset_name == 'vlgcbm_paper_cub':
        ds_base = 'cub'
    else:
        ds_base = dataset_name
    gt_data = GenericDataset(ds_base, split = 'train')
    index = 0
    all_preds = []
    all_gts = []
    for sample in tqdm(loader):
        img,gt_concepts,gt_labels = gt_data[index]
        
        img2,pred_concepts,pred_labels = sample
        pred_concepts = pred_concepts.squeeze().cpu().long()
        if isinstance(gt_concepts, torch.Tensor):
            gt_concepts = gt_concepts.cpu()
        else:
            gt_concepts = torch.tensor(gt_concepts).long()
        #print(sample)
        try:
            if index >= 6 and index <= 6:
                img.save(f"./scripts/img_{index}.png")
                img2_pil = torchvision.transforms.functional.to_pil_image(img2.squeeze(dim=0))
                img2_pil.save(f"./scripts/img2_{index}.png")
                print(pred_concepts, pred_concepts.shape)
                print(gt_concepts, gt_concepts.shape)
        except:
            pass
        
        all_preds.append(pred_concepts)
        all_gts.append(gt_concepts)
        index += 1
        
    all_preds = torch.stack(all_preds, dim=0)
    all_gts = torch.stack(all_gts, dim=0)
    print(all_preds.shape)
    print(all_gts.shape)
    if dataset == 'shapes3d':
        macro_avg_precision = []
        macro_avg_recall = []
        micro_avg_precision = []
        micro_avg_recall = []
        avg_f1 = []
        concept_groups = [10, 10, 10, 8, 4]  # 10 for wall color, 10 background color, 10 object color, 8 sizes and 4 shapes
        start = 0
        for size in concept_groups:
            chunk_gt = all_gts[:,start:start + size]  # Extract the chunk
            chunk_preds = all_preds[:,start:start + size]  # Extract the chunk
            report = classification_report(chunk_gt,chunk_preds, output_dict=True)
            print("a",chunk_gt[0:2])
            print("b",chunk_preds[0:2])
            #print(chunk_preds[0])
            #print(report)
            start += size  # Move to the next chunk
            macro_avg_precision.append(report['macro avg']['precision'])
            macro_avg_recall.append(report['macro avg']['recall'])
            micro_avg_precision.append(report['weighted avg']['precision'])
            micro_avg_recall.append(report['weighted avg']['recall'])
            avg_f1.append(report['macro avg']['f1-score'])
        
        print(f"Annotator {name}:")
        print("Macro average Precision:",np.mean(macro_avg_precision), f"STD:{np.std(macro_avg_precision)}","  Max:", np.max(macro_avg_precision), "  Min:", np.min(macro_avg_precision))
        print("Macro average Recall:",np.mean(macro_avg_recall), f"STD:{np.std(macro_avg_precision)}","  Max:", np.max(macro_avg_recall), "  Min:", np.min(macro_avg_recall))
        print("Micro average Precision:",np.mean(micro_avg_precision), "  Max:", np.max(micro_avg_precision), "  Min:", np.min(micro_avg_precision))
        print("Micro average Recall:",np.mean(micro_avg_recall), "  Max:", np.max(micro_avg_recall), "  Min:", np.min(micro_avg_recall))
        print(f"F1 {np.mean(avg_f1)}")
        print("\n")
    else:
        macro_avg_precision = []
        macro_avg_recall = []
        micro_avg_precision = []
        micro_avg_recall = []
        avg_f1 = []
        print(classification_report(all_gts,all_preds))
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
        print("Macro average Precision:",np.mean(macro_avg_precision), f"STD:{np.std(macro_avg_precision)}","  Max:", np.max(macro_avg_precision), "  Min:", np.min(macro_avg_precision))
        print("Macro average Recall:",np.mean(macro_avg_recall), f"STD:{np.std(macro_avg_precision)}","  Max:", np.max(macro_avg_recall), "  Min:", np.min(macro_avg_recall))
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
        preprocess=transforms.ToTensor(),#backbone.preprocess,
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
            preprocess=transforms.ToTensor(),
            val_split=0.0,
            batch_size=1,
            num_workers=1,
            shuffle=False, 
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
    if dataset=='celeba':
        used_indexes = [25000,50000]
    if dataset == 'shapes3d':
        used_indexes = [0,48000]
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
    if dataset == 'shapes3d':
        c_set = os.path.join(concept_set,'shapes3d.txt')
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
    '''
    compute(loader, dataset, n_concepts=n_concepts, name='CLIP - raw')

    #----------------- Training one param
    cf = copy.deepcopy(clip_features)
    targets = []
    for i in range(len(original_ds)):
        _,concepts,_ = original_ds[i]
        if not isinstance(concepts, torch.Tensor):
            concepts = torch.tensor(concepts)
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
    '''
    #----------------- Training one param for each concept
    cf = copy.deepcopy(clip_features)
    targets = []
    for i in range(len(original_ds)):
        _,concepts,_ = original_ds[i]
        if not isinstance(concepts, torch.Tensor):
            concepts = torch.tensor(concepts)
        targets.append(concepts)
    targets = torch.stack(targets, dim=0)
    logger.info("Training Logistic Regression on All Concepts")
    if dataset=='shapes3d':
        W,B = train_LR_on_concepts_shapes3d(cf, targets)
        print(W)
    else:
        W,B = train_LR_on_concepts(cf, targets)
    
    cf *= W
    cf += B
    #
    
    if dataset == 'shapes3d':
        # Define chunk sizes (must sum to the tensor length)
        concept_groups = [10, 10, 10, 8, 4]  # 10 for wall color, 10 background color, 10 object color, 8 sizes and 4 shapes

        # Compute argmax for each chunk
        argmax_indices = []
        start = 0
        for size in concept_groups:
            chunk = cf[:,start:start + size]  # Extract the chunk
            probs = torch.nn.functional.softmax(chunk, dim=1)
            argmax_indices.append(torch.argmax(chunk,dim=1))  # Compute argmax and store it
            print(argmax_indices)
            start += size  # Move to the next chunk
        preds = torch.zeros(targets.shape)
        start = 0
        print(probs[0])
        
        preds = []
        for sample in range(len(targets)):
            one_hot_concepts = []
            # Construct the predicted concepts constrained on having only one active per concept group
            for i, size in enumerate(concept_groups):
                one_hot = torch.eye(size)
                id = argmax_indices[i][sample]
                #print(one_hot[id])
                one_hot_concepts.append(one_hot[id])
            one_hot_concepts = torch.cat(one_hot_concepts, dim=0)
            preds.append(one_hot_concepts)
            #print(one_hot_concepts)
            #break
        preds = torch.stack(preds, dim=0)
        print(preds[0])
        print(targets[0])
        
    
    else:      
        probs = torch.nn.functional.sigmoid(cf)      
        preds = (probs > 0.5)
    clip_ds = scripts.utils.ClipDataset(preds.long())
    loader = torch.utils.data.DataLoader(clip_ds,batch_size=1,shuffle=False)
    original_ds = GenericDataset(ds_name=dataset, split = 'train')
    n_concepts = original_ds[0][1].shape[0]
    compute(loader, dataset, n_concepts=n_concepts, name='CLIP - LR on all')




if __name__ == '__main__':
    LLava()
    input("Press enter to continue...")
    
    Clip()
    input("Press enter to continue...")
    
    GDino()
    input("Press enter to continue...")
    
    
    