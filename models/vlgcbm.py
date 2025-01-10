import torch.nn as nn
import torch
from models.base import BaseModel
import os 
import json
from loguru import logger
from tqdm import tqdm
import utils.clip as clip
from utils.load_args import load_args
import torchvision.transforms as transforms
from functools import partial

class _Model(torch.nn.Module):
    def __init__(self, backbone_name, W_c, W_g, b_g, proj_mean, proj_std, args, device="cuda"): #backbone_name, W_c, W_g, b_g, proj_mean, proj_std, device="cuda"):
        super().__init__()
        self.args = load_args(args)
        model, _ = get_target_model(backbone_name, device)
        #remove final fully connected layer
        if "clip" in backbone_name:
            self.backbone = model
        elif "cub" in backbone_name:
            self.backbone = partial(get_backbone_function, model)
        else:
            self.backbone = torch.nn.Sequential(*list(model.children())[:-1])
            
        self.proj_layer = torch.nn.Linear(in_features=W_c.shape[1], out_features=W_c.shape[0], bias=False).to(device)
        self.proj_layer.load_state_dict({"weight":W_c})
            
        self.proj_mean = proj_mean
        self.proj_std = proj_std
        
        self.final = torch.nn.Linear(in_features = W_g.shape[1], out_features=W_g.shape[0]).to(device)
        self.final.load_state_dict({"weight":W_g, "bias":b_g})
        self.concepts = None
        
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.proj_layer(x)
        concepts = x
        proj_c = (x-self.proj_mean)/self.proj_std
        x = self.final(proj_c)
        out_dict = {'unnormalized_concepts':concepts, 'concepts':proj_c, 'preds':x}
        return out_dict

    def get_loss(self, args):
        return NotImplementedError('No loss implemented')
        if args.dataset in ['shapes3d', 'dsprites', 'kandinsky','mnist']:
            return CBM_Loss(args, int_C=args.num_C)
        else: 
            return NotImplementedError('Wrong dataset choice')
        
    def start_optim(self, args):
        self.opt = torch.optim.Adam(self.parameters(), args.lr)


class VLGCBM(BaseModel):
  def __init__(self, args):
    super().__init__(self, args)
    

  def train(self, loader):
    pass

  def get_preprocess(self):
    t = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224,224)),
            ]
        )
    return t
    