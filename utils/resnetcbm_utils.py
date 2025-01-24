from loguru import logger
import torch
from tqdm import tqdm
from datasets import get_dataset
from utils.transforms import transform_basic
from loguru import logger

def get_activations_and_targets(model,dataset_name,split,args):
    logger.debug(f'Retrieving labels of {dataset_name} {split}...')
    data = get_dataset(ds_name=dataset_name,split=split,transform=transform_basic)
    n_examples = len(data)
    targets = []
    concepts = [] 
    gt_concetps = []
    model.eval().to(args.device)
    for i in tqdm(range(len(data)), desc='Running model'):
        img, c, label = data[i]
        img = img.to(args.device)
        with torch.no_grad():
            output = model(img.unsqueeze(0))
        targets.append(label.cpu())
        concepts.append(output.cpu())
        gt_concetps.append(c.cpu())
    
    targets = torch.stack(targets, dim=0)
    concepts = torch.cat(concepts, dim=0)
    gt_concetps = torch.stack(gt_concetps, dim=0)
    logger.debug(f"Targets shape: {targets.shape}, Concepts shape: {concepts.shape}, GT Concepts shape: {gt_concetps.shape}")

    return {'concepts':concepts,'targets':targets,'gt_concepts':gt_concetps, 'n_examples':n_examples}