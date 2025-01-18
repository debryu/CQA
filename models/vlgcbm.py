import torch.nn as nn
import torch
from models.base import BaseModel
import os 
import json
from loguru import logger
from tqdm import tqdm
import utils.clip as clip
from utils.args_utils import load_args
import torchvision.transforms as transforms
from utils.vlgcbm_utils import get_target_model, Backbone, BackboneCLIP, ConceptLayer, NormalizationLayer, FinalLayer
from functools import partial

class _Model(torch.nn.Module):
    def __init__(
        self, backbone, cbl, norm, final, args
    ):
        super().__init__()
        self.args = load_args(args)
        self.backbone = backbone
        self.cbl = cbl
        self.normalization = norm
        self.final_layer = final

    def forward(self, x):
        embeddings = self.backbone(x)
        concept_logits = self.cbl(embeddings)
        concept_probs = self.normalization(concept_logits)
        logits = self.final_layer(concept_probs)
        out_dict = {'unnormalized_concepts':concept_logits, 'concepts':concept_probs, 'preds':logits}
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
    self.load_dir = args.load_dir
    # Load Backbone model
    if args.backbone.startswith("clip_"):
        backbone = BackboneCLIP(args.backbone, use_penultimate=args.use_clip_penultimate, device=args.device)
    else:
        backbone = Backbone(args.backbone, args.feature_layer, args.device)
    if os.path.exists(os.path.join(args.load_dir, "backbone.pt")):
        ckpt = torch.load(os.path.join(args.load_dir, "backbone.pt"))
        backbone.backbone.load_state_dict(ckpt)

    # load concepts set directly from load model
    with open(os.path.join(args.load_dir, "concepts.txt"), 'r') as f:
        concepts = f.read().split("\n")
    
    # get model
    cbl = ConceptLayer.from_pretrained(args.load_dir, args.device)
    normalization_layer = NormalizationLayer.from_pretrained(args.load_dir, args.device)
    final_layer = FinalLayer.from_pretrained(args.load_dir, args.device)
    self.model = _Model(backbone, cbl, normalization_layer, final_layer, args)
    
    # Update the args
    self.args = self.model.args
    self.args.batch_size = self.args.cbl_batch_size

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
    