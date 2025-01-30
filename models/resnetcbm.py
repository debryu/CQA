import torch
from models.base import BaseModel
import os 
from loguru import logger
from tqdm import tqdm
import torchvision.transforms as transforms
from functools import partial
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from utils.resnetcbm_utils import PretrainedResNetModel

def get_backbone_function(model, x):
    return model.features(x)

class _Model(torch.nn.Module):
    def __init__(self, args): #backbone_name, W_c, W_g, b_g, proj_mean, proj_std, device="cuda"):
        super().__init__()
        self.backbone = PretrainedResNetModel(args)
        self.final = torch.nn.Linear(in_features = args.num_c, out_features=args.num_classes).to(args.device)
        self.args = args
        
    def forward(self, x):
        concepts = self.backbone(x) 
        concepts = torch.nn.functional.sigmoid(concepts)
        # Generate random preds with dimension batch_size x 2
        preds = self.final(concepts)
        out_dict = {'unnormalized_concepts':concepts, 'concepts':concepts, 'preds':preds}
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


class RESNETCBM(BaseModel):
    def __init__(self, args):
        super().__init__(self, args)
        # Update the load_dir based on the model
        self.model = _Model(args)
        self.args = self.model.args

    def train(self, loader):
        pass

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

