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

# TODO: 
# 1. Remove args loading since it is already loaded in concept_quality.py

'''
Avoid anonymous functions in the code. They can't be pickled and will cause issues when saving the model.
These two functions are created to avoid the issue.
'''
def target_model_function(model, x):
    return model.encode_image(x).float()
def get_backbone_function(model, x):
    return model.features(x)
'''
'''

def get_target_model(target_name, device):
    if target_name.startswith("clip_"):
        target_name = target_name[5:]
        model, preprocess = clip.load(target_name, device=device)
        
        #target_model = lambda x: model.encode_image(x).float() # This is anonymous and can't be pickled
        # Instead I have created a function target_model that does the same thing
        target_model = partial(target_model_function, model)

    elif target_name == 'resnet18_places': 
        target_model = models.resnet18(pretrained=False, num_classes=365).to(device)
        state_dict = torch.load('data/resnet18_places365.pth.tar')['state_dict']
        new_state_dict = {}
        for key in state_dict:
            if key.startswith('module.'):
                new_state_dict[key[7:]] = state_dict[key]
        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
        
    elif target_name == 'resnet18_cub':
        target_model = ptcv_get_model("resnet18_cub", pretrained=True).to(device)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
    
    elif target_name.endswith("_v2"):
        target_name = target_name[:-3]
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V2".format(target_name_cap))
        target_model = eval("models.{}(weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()
        
    else:
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()
    
    return target_model, preprocess


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


class LFCBM(BaseModel):
  def __init__(self, args):
    super().__init__(self, args)
    # Update the load_dir based on the model
    lfcbm_saved_path = self.saved_models[args.model]
    args.load_dir = os.path.join(lfcbm_saved_path,args.load_dir)
    self.load_dir = args.load_dir

    ''' LOAD CBM '''
    with open(os.path.join(self.load_dir ,"args.txt"), 'r') as f:
        self.args = json.load(f)

    logger.debug(f"{self.args}")

    W_c = torch.load(os.path.join(self.load_dir ,"W_c.pt"), map_location=self.args['device'])
    W_g = torch.load(os.path.join(self.load_dir, "W_g.pt"), map_location=self.args['device'])
    b_g = torch.load(os.path.join(self.load_dir, "b_g.pt"), map_location=self.args['device'])

    proj_mean = torch.load(os.path.join(self.load_dir, "proj_mean.pt"), map_location=self.args['device'])
    proj_std = torch.load(os.path.join(self.load_dir, "proj_std.pt"), map_location=self.args['device'])

    self.model = _Model(self.args['backbone'], W_c, W_g, b_g, proj_mean, proj_std, args, self.args['device'])
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
