import torch.nn as nn
import torch
from models.base import BaseModel
import os 
import json
from loguru import logger
from tqdm import tqdm
from utils.args_utils import load_args
import torchvision.transforms as transforms
from functools import partial
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from models.training.resnetcbm import PretrainedResNetModel

# TODO: 
# 1. Remove args loading since it is already loaded in concept_quality.py
# 2. Remove the TEMP comments and workarounds
def get_backbone_function(model, x):
    return model.features(x)

class _Model(torch.nn.Module):
    def __init__(self, args, device="cuda"): #backbone_name, W_c, W_g, b_g, proj_mean, proj_std, device="cuda"):
        super().__init__()
        # Load the PretrainedResNetModel
        # TEMP
        import argparse
        t_args = argparse.Namespace(num_epochs=10, optimizer='adam', optimizer_kwargs={}, scheduler_kwargs={"lr_scheduler_type":"plateau", "additional_kwargs":{}}) 
        t_args.num_c = args.num_c
        self.backbone = PretrainedResNetModel(t_args)
        # Load the backbone
        self.backbone.load_state_dict(torch.load("C:\\Users\\debryu\\Desktop\\VS_CODE\\HOME\\ML\\work\\CQ\\saved_models\\resnetcbm\\cbm_shapes3d_mini_unfrozen1_bestmodel.pth"))
        self.args = args
        self.args.batch_size = 64
        
    def forward(self, x):
        concepts = self.backbone(x) 
        concepts = torch.nn.functional.sigmoid(concepts)
        # Generate random preds with dimension batch_size x 2
        preds = torch.rand((x.shape[0], 2)).to(self.args.device)
        out_dict = {'unnormalized_concepts':concepts, 'concepts':concepts, 'preds':preds}
        return out_dict

    def get_loss(self, args):
        return NotImplementedError('No loss implemented')
        
    def start_optim(self, args):
        return NotImplementedError('No loss implemented')
        self.opt = torch.optim.Adam(self.parameters(), args.lr)


class RESNETCBM(BaseModel):
    def __init__(self, args):
        super().__init__(self, args)
        # Update the load_dir based on the model
        #lfcbm_saved_path = self.saved_models[args.model]
        #args.load_dir = os.path.join(lfcbm_saved_path,args.load_dir)
        self.load_dir = args.load_dir
        self.model = _Model(args)
        self.args = self.model.args

    def train(self, loader):
        pass

    def get_preprocess(self):
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

    def test(self, loader):
        acc_mean = 0.0
        device = self.args.device
        for features, concept_one_hot, targets in tqdm(loader):
            features = features.to(device)
            concept_one_hot = concept_one_hot.to(device)
            targets = targets.to(device)

            # forward pass
            with torch.no_grad():
                out_dict = self.model(features)
                logits = out_dict['preds']

            # calculate accuracy
            preds = logits.argmax(dim=1)
            accuracy = (preds == targets).sum().item()
            acc_mean += accuracy

        return acc_mean / len(loader.dataset)

