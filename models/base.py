from config import SAVED_MODELS_FOLDER
from torch.utils.data import DataLoader
from datasets import get_dataset
from tqdm import tqdm
import torch

class BaseModel():
    def __init__(self,*args):
      self.saved_models = SAVED_MODELS_FOLDER

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.__dict__}>"

    def get_preprocess(self):
      return None
    def get_loader(self, split = 'test'):
      transform = self.get_preprocess()
      dataset_name = self.args.dataset
      data = get_dataset(dataset_name, split = split, transform = transform)
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
      if not hasattr(self, 'load_dir'):
        raise ValueError("load_dir is not defined. Make sure to set it in the custom __init__ function.")
      try:
        batch_size = self.args.batch_size
      except:
        raise ValueError("batch_size is not defined. Make sure to set it in the custom __init__ function.")
      # Check if the methods exist
      
    def run(self, split = 'test'):
      self.check_integrity()
      loader = self.get_loader(split)
      device = self.args.device
      annotations = []
      concepts = []
      labels = []
      preds = []
      acc_mean = 0.0
      debug_i = 0
      for features, concepts_one_hot, targets in tqdm(loader):
        features = features.to(device)
        concepts_one_hot = concepts_one_hot.to(device)
        targets = targets.to(device)

        # forward pass
        with torch.no_grad():
            out_dict = self.model(features)
            logits = out_dict['preds']
            c_repres = out_dict['concepts']
        
        #print(torch.min(c_repres))
        #print(torch.max(c_repres))
        # collect data
        annotations.append(concepts_one_hot)
        concepts.append(c_repres)
        labels.append(targets)
        preds.append(logits)  
        
        # calculate accuracy
        y_preds = logits.argmax(dim=1)
        accuracy = (y_preds == targets).sum().item()
        acc_mean += accuracy
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