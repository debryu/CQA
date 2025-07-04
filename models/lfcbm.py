import torch.nn as nn
import torch
from CQA.models.base import BaseModel
import os 
import json
from loguru import logger
from tqdm import tqdm
from CQA.utils.lfcbm_utils import get_target_model
import torchvision.transforms as transforms
from functools import partial
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def get_backbone_function(model, x):
    return model.features(x)

class _Model(torch.nn.Module):
    def __init__(self, backbone_name, W_c, W_g, b_g, proj_mean, proj_std, args, device="cuda"): #backbone_name, W_c, W_g, b_g, proj_mean, proj_std, device="cuda"):
        super().__init__()
        self.args = args
        logger.debug(f"Saving activations for {args.clip_name} on {backbone_name} backbone")
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

    def load(self):
        # The model is already loaded when initialized
        pass
        
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


class LFCBM(BaseModel):
    def __init__(self, args):
        super().__init__(self, args)
        # Update the load_dir based on the model
        #lfcbm_saved_path = self.saved_models[args.model]
        #args.load_dir = os.path.join(lfcbm_saved_path,args.load_dir)
        self.load_dir = args.load_dir
        self.args = args
        logger.debug(f"{self.args}")

        W_c = torch.load(os.path.join(self.load_dir ,"W_c.pt"), map_location=self.args.device, weights_only=True)
        W_g = torch.load(os.path.join(self.load_dir, "W_g.pt"), map_location=self.args.device, weights_only=True)
        b_g = torch.load(os.path.join(self.load_dir, "b_g.pt"), map_location=self.args.device, weights_only=True)

        proj_mean = torch.load(os.path.join(self.load_dir, "proj_mean.pt"), map_location=self.args.device, weights_only=True)
        proj_std = torch.load(os.path.join(self.load_dir, "proj_std.pt"), map_location=self.args.device, weights_only=True)

        self.model = _Model(self.args.backbone, W_c, W_g, b_g, proj_mean, proj_std, args, self.args.device)

    def train(self, loader):
        pass
    
    def get_transform(self,split):
        # Must use the same transform used for training
        import CQA.utils.clip as clip
        _, preprocess = clip.load(self.args.clip_name, device=self.args.device)
        return preprocess

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

