import os
from loguru import logger
import torch
import copy
import random
from torch.utils.data import DataLoader, TensorDataset
from models.glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from tqdm import tqdm
import torchvision.transforms as transforms
import numpy as np
from datasets import get_dataset, classes
from config import folder_naming_convention, ACTIVATIONS_PATH, CONCEPT_SETS, LLM_GENERATED_ANNOTATIONS
from utils.llamaoracle_utils import query_llama
from models.training.resnetcbm import train as train_cbm

def create_or_load_oracle_ds(args):
    ds = args.dataset.split("_")[0]
    start = args.start_idx
    end = args.end_idx
    # Train
    if os.path.exists(os.path.join(LLM_GENERATED_ANNOTATIONS,f"{args.dataset}_{args.start_idx}-{args.end_idx}_train.pth")):
        train_concepts = torch.load(os.path.join(LLM_GENERATED_ANNOTATIONS,f"{args.dataset}_{args.start_idx}-{args.end_idx}_train.pth"), weights_only=True)
    else:
        logger.info(f"{os.path.join(LLM_GENERATED_ANNOTATIONS,f"{args.dataset}_{args.start_idx}-{args.end_idx}_train.pth")} not found!")
        logger.info(f"Querying train split")
        # load concepts 
        ds = args.dataset.split("_")[0]
        path = CONCEPT_SETS[ds]
        with open(path, 'r') as f:
            concepts = f.read().split("\n")
        queries = []
        if ds == "celeba":
            for c in concepts:
                queries.append(f"a person with {c}")
        if ds == "shapes3d":
            for c in concepts:
                queries.append(f"a {c}")
        print(queries)
        t = transforms.Compose([
        transforms.Lambda(lambda x: np.array(x))  # Ensure it's a NumPy array
        ])
        
        ds = get_dataset(args.dataset, subset_indices=[start,end], split='train', transform=t)
        dl = DataLoader(ds, batch_size=1, shuffle=False)
        train_concepts = query_llama(dl, queries)
        torch.save(train_concepts, os.path.join(LLM_GENERATED_ANNOTATIONS,f"{args.dataset}_{args.start_idx}-{args.end_idx}_train.pth"))
    # Val
    if os.path.exists(os.path.join(LLM_GENERATED_ANNOTATIONS,f"{args.dataset}_{args.start_idx}-{args.end_idx}_val.pth")):
        val_concepts = torch.load(os.path.join(LLM_GENERATED_ANNOTATIONS,f"{args.dataset}_{args.start_idx}-{args.end_idx}_val.pth"), weights_only=True)
    else:
        # load concepts 
        logger.info(f"Querying val split")
        ds = args.dataset.split("_")[0]
        path = CONCEPT_SETS[ds]
        with open(path, 'r') as f:
            concepts = f.read().split("\n")
        queries = []
        for c in concepts:
            queries.append(f"a person with {c}")
        logger.debug(queries)
        t = transforms.Compose([
        transforms.Lambda(lambda x: np.array(x))  # Ensure it's a NumPy array
        ])
        
        ds = get_dataset(args.dataset, subset_indices=[start,end], split='val', transform=t)
        dl = DataLoader(ds, batch_size=1, shuffle=False)
        val_concepts = query_llama(dl, queries)
        torch.save(val_concepts, os.path.join(LLM_GENERATED_ANNOTATIONS,f"{args.dataset}_{args.start_idx}-{args.end_idx}_val.pth"))
    
    return {"train":train_concepts, "val":val_concepts, "test":None}
    # Skip test for now, we don't need it since we test it with the real labels
    # Test
    if os.path.exists(os.path.join(LLM_GENERATED_ANNOTATIONS,f"{args.dataset}_{args.start_idx}-{args.end_idx}_test.pth")):
        test_concepts = torch.load(os.path.join(LLM_GENERATED_ANNOTATIONS,f"{args.dataset}_{args.start_idx}-{argsend_idx}_test.pth"))
    else:
        # load concepts 
        logger.info(f"Querying test split")
        ds = args.dataset.split("_")[0]
        path = CONCEPT_SETS[ds]
        with open(path, 'r') as f:
            concepts = f.read().split("\n")
        queries = []
        for c in concepts:
            queries.append(f"a person with {c}")
        print(queries)
        t = transforms.Compose([
        transforms.Lambda(lambda x: np.array(x))  # Ensure it's a NumPy array
        ])
        
        ds = get_dataset(args.dataset, subset_indices=[start,end], split='train', transform=t)
        dl = DataLoader(ds, batch_size=1, shuffle=False)
        test_concepts = query_llama(dl, queries)
        torch.save(test_concepts, os.path.join(LLM_GENERATED_ANNOTATIONS,f"{args.dataset}_{args.start_idx}-{args.end_idx}_test.pth"))
    return {"train":train_concepts, "val":val_concepts, "test":test_concepts}

def train(args):
    ds = args.dataset.split('_')[0]
    # Due to time consuming task of labeling with llama, make by default to use the mini version of the dataset
    # The number of samples can still be changed with arguments start_idx and end_idx
    args.dataset = f"{ds}_mini"
    llama_concepts = create_or_load_oracle_ds(args)
    args.num_c = llama_concepts['train'].shape[1]
    
    # This will not persist, as it is created runtime
    class LlamaAnnotatedDataset(torch.utils.data.Dataset):
        # Store some variables
        temp_args = args
        temp_ds = ds
        def __init__(self,**kwargs):
            self.ds_name = self.temp_ds + "_mini"
            split = kwargs.get('split')
            self.has_concepts = True
            super().__init__()
            self.original_ds = get_dataset(self.ds_name, subset_indices=[self.temp_args.start_idx,self.temp_args.end_idx],**kwargs)
            
            if split == 'train':
                self.llama_concepts = llama_concepts['train']
            elif split == 'val' or split == 'valid':
                self.llama_concepts = llama_concepts['val']
            assert len(self.original_ds)==len(self.llama_concepts)
        
        def __len__(self):
            return len(self.original_ds)
        
        def __getitem__(self, index: int):
            x,c,y = self.original_ds[index]
            llama_c = self.llama_concepts[index]
            return x, llama_c, y

    # Inject this dataset into the available datasets temporary
    logger.debug("Injecting dataset")
    new_temp_args = copy.deepcopy(args)
    new_temp_args.dataset = f'{args.dataset}_temp'
    classes[new_temp_args.dataset] = LlamaAnnotatedDataset
    logger.debug(f"Available datasets: {classes}")
    train_cbm(new_temp_args)
    
    return args

