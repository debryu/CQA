import torchvision.transforms as transforms
import torch
from functools import partial
import os 
import json
from loguru import logger
from tqdm import tqdm
import CQA.utils.clip as clip
from CQA.utils.args_utils import load_args
from CQA.models.base import BaseModel

class _Model(torch.nn.Module):
    def __init__(
        ..., args
    ):
        super().__init__()
        self.args = load_args(args)

    def forward(self, x):
        unnorm_concepts = 
        concepts = 
        predictions = 
        out_dict = {'unnormalized_concepts':unnorm_concepts, 'concepts':concepts, 'preds':predictions}
        return out_dict

class TEMPLATE(BaseModel):
    def __init__(self, args):
        super().__init__(self, args)
        self.load_dir = args.load_dir
        
        # Call the model
        self.model = _Model(..., args)
        
        # Update the args
        self.args = args



    