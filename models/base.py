from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from loguru import logger
from torchvision import transforms
from torchvision.transforms.v2 import ColorJitter, GaussianNoise

from config import SAVED_MODELS_FOLDER
from datasets import GenericDataset

class BaseModel():
    def __init__(self,*args):
      self.saved_models = SAVED_MODELS_FOLDER

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.__dict__}>"

    def get_transform(self, split):
      logger.debug(f"Using default method get_transform for {split}")
      if split == 'train':  
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            #transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            ColorJitter(brightness=0.3, contrast=0.5, saturation=0.1, hue=0.0),
            GaussianNoise(0,0.02),
        ])
      else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
        ])

    def get_loader(self, split):
      logger.debug(f"Using default method get_loader for {split}")
      transform = self.get_transform(split=split)
      dataset_name = self.args.dataset
      data = GenericDataset(dataset_name, split = split, transform = transform)
      if split == 'train':
        return DataLoader(data, batch_size = self.args.batch_size, shuffle = True)
      else:
        return DataLoader(data, batch_size = self.args.batch_size, shuffle = False)
        
    def check_integrity(self):
      # Check if the variable exists
      if not hasattr(self, 'args'):
        raise ValueError("args is not defined. Make sure to set it in the custom __init__ function.")
      if not hasattr(self, 'model'):
        raise ValueError("model is not defined. Make sure to set it in the custom __init__ function.")
      try:
        batch_size = self.args.batch_size
      except:
        raise ValueError("batch_size is not defined. Make sure to set it in the custom __init__ function.")
      # Check if the methods exist
      
    def run(self, split = 'test'):
      logger.debug(f"Running model on {split} split.")
      self.check_integrity()
      loader = self.get_loader(split)
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
            #print(c_repres)
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

    def update_args(self):
      self.args = self.model.args  
      return self.args