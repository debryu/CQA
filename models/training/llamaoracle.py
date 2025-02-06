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
from utils.llamaoracle_utils import query_llama, unify_pickles
from models.training.resnetcbm import train as train_cbm

def check_content(folder, indexes):
    logger.debug(f"Checking content of {folder} from {indexes[0]} to {indexes[1]}")
    missing = list(range(indexes[0], indexes[1]))
    for file in tqdm(os.listdir(folder), desc="Checking folder content"):
        id = int(file.split("_")[-1].split(".")[0])
        #print(id) 
        #print(missing)
        if id in missing:
            missing.remove(id)
    
    if len(missing) > 0:
        for m in missing:
            logger.info(f"Missing query_{m}.pkl from {folder}")
    return missing


def create_or_load_oracle_ds(args):
    start = args.start_idx
    end = args.end_idx
    concepts_dict = {}
    for split in ['val','train']: #,'test']:
        os.makedirs(LLM_GENERATED_ANNOTATIONS, exist_ok=True)
        os.makedirs(os.path.join(LLM_GENERATED_ANNOTATIONS, args.dataset), exist_ok=True)
        os.makedirs(os.path.join(f"{LLM_GENERATED_ANNOTATIONS}/{args.dataset}",split), exist_ok=True)
        current_folder = os.path.join(f"{LLM_GENERATED_ANNOTATIONS}/{args.dataset}",split)

        # Get the original dataset
        
        t = transforms.Compose([
        transforms.Lambda(lambda x: np.array(x))  # Ensure it's a NumPy array
        ])
        ds_name = args.dataset.split("_")[0]
        ds = get_dataset(ds_name, split=split, transform=t)
        if start is None:
            start = 0
        if end is None:
            end = len(ds)  # One more just to be sure
        
        # Celeba train is the only exception since we have a weird train (from 25000 to 50000 instead of the beginning)
        if args.dataset == 'celeba' and split == 'train':
            used_indexes = [25000,50000]
        else:
            used_indexes = [0,len(ds)]

        logger.debug(f"Used indexes: {used_indexes}")
        missing = []
        # Check if the dataset exists
        logger.debug(f"Checking if {os.path.join(LLM_GENERATED_ANNOTATIONS,f'{args.dataset}_{split}_{used_indexes[0]}_{used_indexes[1]}.pth')} exists")
        if os.path.exists(os.path.join(LLM_GENERATED_ANNOTATIONS,f"{args.dataset}_{split}_{used_indexes[0]}_{used_indexes[1]}.pth")):
            # Load
            logger.debug(f"Loading {os.path.join(LLM_GENERATED_ANNOTATIONS,f"{args.dataset}_{split}_{used_indexes[0]}_{used_indexes[1]}.pth")}")
            concepts_dict[split] = torch.load(os.path.join(LLM_GENERATED_ANNOTATIONS,f"{args.dataset}_{split}_{used_indexes[0]}_{used_indexes[1]}.pth"), weights_only=True) 
        else:
            missing = check_content(current_folder, used_indexes)
            if len(missing) == 0:
                logger.debug(f"Unifying inside {os.path.join(LLM_GENERATED_ANNOTATIONS,f'{args.dataset}_{split}.pth')}")
                # Save the concepts in a single .pth file
                concepts_dict[split] = unify_pickles(current_folder, os.path.join(LLM_GENERATED_ANNOTATIONS,f"{args.dataset}_{split}_{used_indexes[0]}_{used_indexes[1]}.pth"), indexes=used_indexes)
            else:
                logger.info(f"Creating or loading oracle dataset for {args.dataset}:{split} from {start} to {end}")
                
                path = CONCEPT_SETS[ds_name]
                with open(path, 'r') as f:
                    concepts = f.read().split("\n")
                queries = []
                if ds_name == 'celeba':
                    for c in concepts:
                        queries.append(f"a person with {c}")
                if ds_name == 'shapes3d':
                    for c in concepts:
                        queries.append(f"a {c}")
                else:
                    for c in concepts:
                        queries.append(c)
                logger.debug(queries)
                
                dl = DataLoader(ds, batch_size=1, shuffle=False)
                llm_concepts = query_llama(dl, queries, os.path.join(f"{LLM_GENERATED_ANNOTATIONS}/{args.dataset}",split), args=args, range=[start,end], missing=missing)
                concepts_dict[split] = unify_pickles(current_folder, os.path.join(LLM_GENERATED_ANNOTATIONS,f"{args.dataset}_{split}.pth"), indexes=used_indexes)
               

    return concepts_dict

def train(args):
    ds = args.dataset.split('_')[0]
    llama_concepts = create_or_load_oracle_ds(args)
    #print(llama_concepts['train'])
    args.num_c = llama_concepts['train'].shape[1]
    
    # This will not persist, as it is created runtime
    class LlamaAnnotatedDataset(torch.utils.data.Dataset):
        # Store some variables
        temp_args = args
        temp_ds = ds
        train_concepts = llama_concepts['train']
        val_concepts = llama_concepts['val']
        def __init__(self,**kwargs):
            split = kwargs.get('split')
            super().__init__()
            self.original_ds = get_dataset(self.temp_ds, **kwargs)
            
            if split == 'train':
                self.llama_concepts = self.train_concepts
            elif split == 'val' or split == 'valid':
                self.llama_concepts = self.val_concepts
            else:
                raise NotImplementedError(f"Split {split} not implemented")
            #print(self.llama_concepts.shape)
            #print(len(self.original_ds))
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
    new_temp_args.dataset = f'{args.dataset}_oracle'
    classes[new_temp_args.dataset] = LlamaAnnotatedDataset
    logger.debug(f"Available datasets: {classes}")
    final_args = train_cbm(new_temp_args)
    vars(args).update(vars(final_args))
    
    return args

