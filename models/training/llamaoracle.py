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

def create_or_load_oracle_ds(args):
    ds = args.dataset.split("_")[0]
    start = args.start_idx
    end = args.end_idx
    concepts_dict = {}
    for split in ['val','train','test']:
        os.makedirs(LLM_GENERATED_ANNOTATIONS, exist_ok=True)
        os.makedirs(os.path.join(LLM_GENERATED_ANNOTATIONS, args.dataset), exist_ok=True)
        os.makedirs(os.path.join(f"{LLM_GENERATED_ANNOTATIONS}/{args.dataset}",split), exist_ok=True)
        current_folder = os.path.join(f"{LLM_GENERATED_ANNOTATIONS}/{args.dataset}",split)

        
        # Get the original dataset
        ds = args.dataset.split("_")[0]
        t = transforms.Compose([
        transforms.Lambda(lambda x: np.array(x))  # Ensure it's a NumPy array
        ])
        original_ds = get_dataset(ds, split=split, transform=t)
        if start is None:
            start = 0
        if end is None:
            end = len(original_ds)  # One more just to be sure
        
        # Check if the dataset exists
        if os.path.exists(os.path.join(LLM_GENERATED_ANNOTATIONS,f"{args.dataset}_{split}.pth")):
            # Load
            logger.debug(f"Loading {os.path.join(LLM_GENERATED_ANNOTATIONS,f'{args.dataset}_{split}.pth')}")
            concepts_dict[split] = torch.load(os.path.join(LLM_GENERATED_ANNOTATIONS,f"{args.dataset}_{split}.pth"), weights_only=True) 
            print(concepts_dict[split].shape)
        elif len(os.listdir(current_folder)) == len(original_ds):
            logger.debug(f"Unifying inside {os.path.join(LLM_GENERATED_ANNOTATIONS,f'{args.dataset}_{split}.pth')}")
            # Save the concepts in a single .pth file
            concepts_dict[split] = unify_pickles(current_folder, os.path.join(LLM_GENERATED_ANNOTATIONS,f"{args.dataset}_{split}.pth"))
        else:
            logger.info(f"Creating or loading oracle dataset for {args.dataset} from {start} to {end}")
            
            path = CONCEPT_SETS[ds]
            with open(path, 'r') as f:
                concepts = f.read().split("\n")
            queries = []
            for c in concepts:
                queries.append(f"a person with {c}")
            logger.debug(queries)
            
            dl = DataLoader(original_ds, batch_size=1, shuffle=False)
            llm_concepts = query_llama(dl, queries, os.path.join(f"{LLM_GENERATED_ANNOTATIONS}/{args.dataset}",split), args=args, range=[start,end])
            
            if len(os.listdir(current_folder)) == len(original_ds):
                concepts_dict[split] = unify_pickles(current_folder, os.path.join(LLM_GENERATED_ANNOTATIONS,f"{args.dataset}_{split}.pth"))
            else:
                logger.error("Not all concepts were queried, please check the folder")
                raise ValueError("Not all concepts were queried, please check the folder")

    return concepts_dict

def train(args):
    ds = args.dataset.split('_')[0]
    llama_concepts = create_or_load_oracle_ds(args)
    print(llama_concepts['train'])
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
    final_args = train_cbm(new_temp_args)
    args.update(final_args)
    
    return args

