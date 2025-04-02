import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from models.base import BaseModel
import os 
import json
from loguru import logger
import torchvision.transforms as transforms
from functools import partial
from tqdm import tqdm
from datasets import GenericDataset
from utils.lfcbm_utils import get_target_model, save_activations, get_save_names
from utils.args_utils import load_args
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from config import ACTIVATIONS_PATH
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def get_backbone_function(model, x):
    return model.features(x)

# TODO: correct transform loader depending on the trainig

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
        self.W_g = W_g
        #self.final = torch.nn.Linear(in_features = W_g.shape[1], out_features=W_g.shape[0]).to(device)
        #self.final.load_state_dict({"weight":W_g, "bias":b_g})
        #self.proj_mean = proj_mean
        #self.proj_std = proj_std
        self.concepts = None
        
        
    def forward(self, x):
        #print(self.clip_features[x])
        concepts = self.clip_features[x]*100
        #print(concepts)
        #print(concepts.shape)
        #input("")
        proj_c = concepts     # Temp
        #print(proj_c)
        #input("")
        #proj_c = (x-self.proj_mean)/self.proj_std
        #print(proj_c[0])
        #print(self.W_g[0])
        #input("")
        mat = F.softmax(self.W_g, dim=-1)
        y = proj_c @ mat.t()
        #print(y)
        #print(self.W_g)
        #input("")
        #print(self.W_g)
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
        
        test_target_save_name, test_clip_save_name, test_text_save_name = get_save_names(args.clip_name, args.backbone, 
                                            args.feature_layer,ds_test, args.concept_set, "avg", ACTIVATIONS_PATH['shared'])
        
        ds_test = f"{args.dataset}_train"
        save_activations(clip_name = args.clip_name, target_name = args.backbone, 
                               target_layers = [args.feature_layer], d_probe = ds_test,
                               concept_set = args.concept_set, batch_size = args.batch_size, 
                               device = args.device, pool_mode = "avg", save_dir = args.activation_dir)
        
        train_target_save_name, train_clip_save_name, train_text_save_name = get_save_names(args.clip_name, args.backbone, 
                                            args.feature_layer,ds_test, args.concept_set, "avg", ACTIVATIONS_PATH['shared'])
        
        ds_test = f"{args.dataset}_val"
        save_activations(clip_name = args.clip_name, target_name = args.backbone, 
                               target_layers = [args.feature_layer], d_probe = ds_test,
                               concept_set = args.concept_set, batch_size = args.batch_size, 
                               device = args.device, pool_mode = "avg", save_dir = args.activation_dir)
        
        val_target_save_name, val_clip_save_name, val_text_save_name = get_save_names(args.clip_name, args.backbone, 
                                            args.feature_layer,ds_test, args.concept_set, "avg", ACTIVATIONS_PATH['shared'])

        #load features
        with torch.no_grad():
            target_features = torch.load(test_target_save_name, map_location="cpu", weights_only=True).float()
            image_features = torch.load(test_clip_save_name, map_location="cpu", weights_only=True).float()
            image_features /= torch.norm(image_features, dim=1, keepdim=True)
            text_features = torch.load(test_text_save_name, map_location="cpu", weights_only=True).float()
            text_features /= torch.norm(text_features, dim=1, keepdim=True)
            
            self.test_clip_features = image_features @ text_features.T            # Namely, P
            del image_features, text_features

            target_features = torch.load(train_target_save_name, map_location="cpu", weights_only=True).float()
            image_features = torch.load(train_clip_save_name, map_location="cpu", weights_only=True).float()
            image_features /= torch.norm(image_features, dim=1, keepdim=True)
            text_features = torch.load(train_text_save_name, map_location="cpu", weights_only=True).float()
            text_features /= torch.norm(text_features, dim=1, keepdim=True)
            
            self.train_clip_features = image_features @ text_features.T            # Namely, P
            del image_features, text_features

            target_features = torch.load(val_target_save_name, map_location="cpu", weights_only=True).float()
            image_features = torch.load(val_clip_save_name, map_location="cpu", weights_only=True).float()
            image_features /= torch.norm(image_features, dim=1, keepdim=True)
            text_features = torch.load(val_text_save_name, map_location="cpu", weights_only=True).float()
            text_features /= torch.norm(text_features, dim=1, keepdim=True)
            
            self.val_clip_features = image_features @ text_features.T            # Namely, P
            del image_features, text_features

        W_g = torch.load(os.path.join(self.load_dir, "W_g.pt"), map_location=self.args.device, weights_only=True)
        b_g = torch.load(os.path.join(self.load_dir, "b_g.pt"), map_location=self.args.device, weights_only=True)
        self._model_args = {
            'backbone_name':args.backbone, 
            'W_g':W_g, 
            'b_g':b_g, 
            'args': args,
        }
        self.model = _Model(args.backbone, self.test_clip_features, W_g, b_g, args, args.device)
        

    def get_loader(self, split='test'):
        transform = self.get_transform(split)
        dataset_name = self.args.dataset
        data = GenericDataset(dataset_name, split = split, transform = transform)
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
        # Must use the same transform used for training
        import utils.clip as clip
        _, preprocess = clip.load(self.args.clip_name, device=self.args.device)
        return preprocess
    
    def run(self, split = 'test'):
      logger.debug(f"Running model on {split} split.")
      self.check_integrity()
      loader = self.get_loader(split)
      if split == 'train':
          clip_features = self.train_clip_features
          self.model = _Model(clip_features=clip_features,**self._model_args)
      elif split == 'val':
          clip_features = self.val_clip_features
          self.model = _Model(clip_features=clip_features,**self._model_args)
      else:
          # Use the ones initialized in __init__
          # self.model and clip features are already initialized for test in init
          pass
      
      self.model.args.transform = str(self.get_transform(split=split))
      self.model.load() # Load the model weights
      device = self.args.device
      annotations = []
      concepts = []
      labels = []
      preds = []
      acc_mean = 0
      debug_i = 0
      n = 0
      for features, concepts_one_hot, targets in tqdm(loader, desc = f"Running {split}"):
        features = features.to(device)
        concepts_one_hot = concepts_one_hot.to(device)
        targets = targets.to(device)

        # forward pass
        with torch.no_grad():
            out_dict = self.model(features)
            logits = out_dict['preds'].float()
            c_repres = out_dict['concepts']
            annotations.append(concepts_one_hot)
            concepts.append(c_repres)
            labels.append(targets)
            preds.append(logits)  
            # calculate accuracy
            y_preds = logits.argmax(dim=1)
            accuracy = (y_preds.to('cpu') == targets.to('cpu')).sum().item()
            acc_mean += accuracy
            
        n += len(targets)
        #if debug_i > 5:
        #  break
        debug_i += 1
      annotations = torch.cat(annotations, dim=0).cpu()
      concepts = torch.cat(concepts, dim=0).cpu()
      labels = torch.cat(labels, dim=0).cpu()
      preds = torch.cat(preds, dim=0).cpu()
      out_dict = {
        "concepts_gt": annotations,
        "concepts_pred": concepts,
        "labels_gt": labels,
        "labels_pred": preds,
        "accuracy": acc_mean / len(loader.dataset)
      }

      return out_dict


