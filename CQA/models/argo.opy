import torch
from CQA.models.base import BaseModel
import os 
from loguru import logger
from tqdm import tqdm
import torchvision.transforms as transforms
from functools import partial
from CQA.datasets import GenericDataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import random
from torch.utils.data import DataLoader
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from CQA.utils.resnetcbm_utils import PretrainedResNetModel

def get_backbone_function(model, x):
    return model.features(x)

class _Model(torch.nn.Module):
    def __init__(self, args): #backbone_name, W_c, W_g, b_g, proj_mean, proj_std, device="cuda"):
        super().__init__()
        #args.fc_layers = []
        self.backbone = PretrainedResNetModel(args)
        self.final = torch.nn.Linear(in_features = args.num_c, out_features=args.num_classes).to(args.device)
        self.args = args
    
    def forward(self, x):
        concepts = self.backbone(x) 
        probs = torch.nn.functional.sigmoid(concepts)
        # Generate random preds with dimension batch_size x 2
        preds = self.final(concepts)
        # The concepts are now the probs since we are approximating a GP
        out_dict = {'unnormalized_concepts':concepts, 'concepts':concepts, 'preds':preds, 'concept_probs':probs}
        return out_dict

    def load(self):
        # Load the backbone
        self.backbone.load_state_dict(torch.load(os.path.join(self.args.load_dir, f'best_backbone_{self.args.model}.pth'), weights_only=True))
        W_g = torch.load(os.path.join(self.args.load_dir, "W_g.pt"), map_location=self.args.device, weights_only=True)
        b_g = torch.load(os.path.join(self.args.load_dir, "b_g.pt"), map_location=self.args.device, weights_only=True)
        self.final.load_state_dict({"weight":W_g, "bias":b_g})
        return 
    
    def get_loss(self, args):
        return NotImplementedError('No loss implemented')
        
    def start_optim(self, args):
        return NotImplementedError('No loss implemented')
        self.opt = torch.optim.Adam(self.parameters(), args.lr)


class ARGO(BaseModel):
    def __init__(self, args):
        super().__init__(self, args)
        # Update the load_dir based on the model
        self.model = _Model(args)
        self.args = self.model.args

    def train(self, loader):
        pass
    
    #@staticmethod
    #def get_transform(split):
    #    logger.debug(f"Using argo method get_transform for {split}")
    #    return transforms.Compose([
    #        transforms.ToTensor(),
    #        transforms.Resize((224,224)),
    #    ])
    
    def get_loader(self, split):
        dataset_name = self.args.dataset
        dataset_base = dataset_name.split("_")[0]
        transform = self.get_transform(split=split)
        gt_data = GenericDataset(dataset_base, split = split, transform = transform)
        return DataLoader(gt_data, batch_size=self.args.batch_size, shuffle=False)
    
    def get_subset_loader(self, split, val_split_count = 1000):
        if split not in ['val']:
            raise ValueError()
        
        dataset_name = self.args.dataset
        transform = self.get_transform(split=split)
        gt_data = GenericDataset(dataset_name, split = split, transform = transform)
        self.subset_indices = random.sample(range(len(gt_data)), val_split_count)
        gt_data_subset = torch.utils.data.Subset(gt_data, self.subset_indices)
        self.val_data = gt_data_subset
        logger.warning(f"Using subset val data with {val_split_count} samples")
        return DataLoader(gt_data_subset, batch_size=self.args.batch_size, shuffle=False)
    '''
    def get_transform(self):
        t = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((224,224)),
                ]
            )
        c = Compose([
                Resize((224,224), interpolation=BICUBIC),
                CenterCrop((224,224)),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

        return t
    '''

