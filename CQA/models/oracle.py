import torchvision.transforms as transforms
import torch
from functools import partial
import os 
import json
from loguru import logger
from tqdm import tqdm
from CQA.models.base import BaseModel
from CQA.models.resnetcbm import _Model as ResNet_Model
from torch.utils.data import DataLoader
import CQA.scripts.utils as scripts
from CQA.datasets import GenericDataset
from CQA.config import LLM_GENERATED_ANNOTATIONS
from CQA.config import SPLIT_INDEXES


class ORACLE(BaseModel):
    def __init__(self, args):
        super().__init__(self, args)
        self.load_dir = args.load_dir
        # Call the model
        self.model = ResNet_Model(args)
        # Update the args
        self.args = args

    # Load the LLM-annotated dataset when the split is train or val, otherwise load ground truth dataset for test evaluations.
    def get_loader(self, split):
        dataset_name = self.args.dataset
        dataset_base = dataset_name.split("_")[0]
        transform = self.get_transform(split=split)
        gt_data = GenericDataset(dataset_base, split = split, transform = transform)
        #if split == 'train':
        #    return DataLoader(data, batch_size = self.args.batch_size, shuffle = True)
        #else:
        #    return DataLoader(data, batch_size = self.args.batch_size, shuffle = False)
        if split == 'test':
            logger.debug(f"Using default method get_loader for {split}")  
            data = GenericDataset(dataset_base, split = split, transform = transform)
            return DataLoader(data, batch_size = self.args.batch_size, shuffle = False)
        else:
            gt_data = GenericDataset(ds_name=f"{dataset_base}", split=split, transform=transform)
            used_indexes = SPLIT_INDEXES[f"{dataset_base}_{split}"]
            concepts = torch.load(os.path.join(LLM_GENERATED_ANNOTATIONS,f"{dataset_base}_{split}_{used_indexes[0]}_{used_indexes[1]}.pth"), weights_only=True) 
            args = {"original_ds":gt_data, "train_concepts":concepts}
            oracle_data = scripts.AnnotatedDataset(**args)
            if split == 'train':
                return DataLoader(oracle_data, batch_size = self.args.batch_size, shuffle = True)
            else:
                return DataLoader(oracle_data, batch_size = self.args.batch_size, shuffle = False)
      
        