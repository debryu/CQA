import torchvision.transforms as transforms
import torch
from functools import partial
import os 
import json
from loguru import logger
from tqdm import tqdm
import utils.clip as clip
from utils.args_utils import load_args
from models.base import BaseModel
from models.resnetcbm import _Model as ResNet_Model

class LLAMAORACLE(BaseModel):
    def __init__(self, args):
        super().__init__(self, args)
        self.load_dir = args.load_dir
        # Call the model
        self.model = ResNet_Model(args)
        
        # Update the args
        self.args = args

    # Define preprocess transform
    def get_transform(self):
        t = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((224,224)),
                ]
            )
        return t
    