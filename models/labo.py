import torch
from torch.utils.data import Dataset, DataLoader
from models.base import BaseModel
import os 
import json
from loguru import logger
import torchvision.transforms as transforms
from functools import partial
from tqdm import tqdm
from datasets import get_dataset
from utils.lfcbm_utils import get_target_model, save_activations, get_save_names
from utils.args_utils import load_args
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
    def __init__(self, backbone_name, clip_features, W_g, b_g, args, device="cuda"): #backbone_name, W_c, W_g, b_g, proj_mean, proj_std, device="cuda"):
        super().__init__()
        model, _ = get_target_model(backbone_name, device)
        self.args = args
        #remove final fully connected layer
        if "clip" in backbone_name:
            self.backbone = model
        elif "cub" in backbone_name:
            self.backbone = partial(get_backbone_function, model)
        else:
            self.backbone = torch.nn.Sequential(*list(model.children())[:-1])
        self.clip_features = clip_features.to(device)
        self.final = torch.nn.Linear(in_features = W_g.shape[1], out_features=W_g.shape[0]).to(device)
        self.final.load_state_dict({"weight":W_g, "bias":b_g})
        self.concepts = None
        
    def forward(self, x):
        concepts = self.clip_features[x]
        proj_c = concepts   # Temp
        y = self.final(proj_c)
        out_dict = {'unnormalized_concepts':concepts, 'concepts':proj_c, 'preds':y}
        return out_dict

    def get_loss(self, args):
        return NotImplementedError('No loss implemented')
        if args.dataset in ['shapes3d', 'dsprites', 'kandinsky','mnist']:
            return CBM_Loss(args, int_C=args.num_C)
        else: 
            return NotImplementedError('Wrong dataset choice')
        
    def start_optim(self, args):
        self.opt = torch.optim.Adam(self.parameters(), args.lr)

    def load(self):
        # The model is already loading in init
        pass


class LABO(BaseModel):
    def __init__(self, args):
        super().__init__(self, args)
        # Update the load_dir based on the model
        #lfcbm_saved_path = self.saved_models[args.model]
        #args.load_dir = os.path.join(lfcbm_saved_path,args.load_dir)
        self.load_dir = args.load_dir
        self.args = args
        logger.debug(f"{self.args}")
        # Save test clip activations
        #save activations and get save_paths
        ds_test = f"{args.dataset}_test"
        save_activations(clip_name = args.clip_name, target_name = args.backbone, 
                               target_layers = [args.feature_layer], d_probe = ds_test,
                               concept_set = args.concept_set, batch_size = args.batch_size, 
                               device = args.device, pool_mode = "avg", save_dir = args.activation_dir)
        
        target_save_name, clip_save_name, text_save_name = get_save_names(args.clip_name, args.backbone, 
                                            args.feature_layer,ds_test, args.concept_set, "avg", args.activation_dir)

        #load features
        with torch.no_grad():
            target_features = torch.load(target_save_name, map_location="cpu", weights_only=True).float()
            image_features = torch.load(clip_save_name, map_location="cpu", weights_only=True).float()
            #image_features /= torch.norm(image_features, dim=1, keepdim=True)
            text_features = torch.load(text_save_name, map_location="cpu", weights_only=True).float()
            #text_features /= torch.norm(text_features, dim=1, keepdim=True)
            
            clip_features = image_features @ text_features.T            # Namely, P
            del image_features, text_features

        W_g = torch.load(os.path.join(self.load_dir, "W_g.pt"), map_location=self.args.device, weights_only=True)
        b_g = torch.load(os.path.join(self.load_dir, "b_g.pt"), map_location=self.args.device, weights_only=True)
        self.model = _Model(args.backbone, clip_features, W_g, b_g, args, args.device)
        

    def get_loader(self, split='test'):
        transform = self.get_transform(split)
        dataset_name = self.args.dataset
        data = get_dataset(dataset_name, split = split, transform = transform)

        class IndexedDataset(Dataset):
            def __init__(self, data):
                """
                Args:
                    data: A list, tensor, or any iterable of data samples.
                """
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                """
                Args:
                    idx: Index of the data sample.
                Returns:
                    A tuple of (sample, idx).
                """
                img, c, label = self.data[idx]
                # Return the index of the image instead of the image
                # in this way, the model just looks at the CLIP features[idx]
                return idx, c, label
        indexed_data = IndexedDataset(data)
        return DataLoader(indexed_data, batch_size = self.args.batch_size, shuffle = False)


    def get_transform(self,split):
        c = Compose([
                Resize((224,224), interpolation=BICUBIC),
                CenterCrop((224,224)),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        return c


