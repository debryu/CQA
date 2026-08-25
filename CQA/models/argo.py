import torch
from CQA.models.base import BaseModel
import os 
from loguru import logger
from tqdm import tqdm
import torchvision.transforms as transforms
from functools import partial
from CQA.datasets import GenericDataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import random
from torch.utils.data import DataLoader
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from CQA.utils.resnetcbm_utils import PretrainedResNetModel

def get_backbone_function(model, x):
    return model.features(x)

class ArgoConceptExtractor():
    def __init__(self):
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.val_x = None
        self.val_y = None
        
    def set_ds(self, train, val, test):
        self.train_ds = train
        self.val_ds = val
        self.test_ds = test
        
    def load(self, train_x, train_y, val_x, val_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.test_x = test_x
        self.test_y = test_y
        
    def get_concepts_and_labels(self, split):
        if split == 'train':
            concepts = []
            labels = []
            stds = []
            for i in tqdm(range(len(self.train_ds)), desc="loading train"):
                _, preds, std, a, b, l = self.train_ds[i]
                concepts.append(preds)
                stds.append(std)
                labels.append(l)
                
                
        elif split == 'test':
            concepts = []
            labels = []
            stds = []
            for i in tqdm(range(len(self.test_ds)), desc="loading test"):
                _, preds, std, a, b, l = self.train_ds[i]
                concepts.append(preds)
                stds.append(std)
                labels.append(l)
                
                
        elif split == 'val':
            concepts = []
            labels = []
            stds = []
            for i in tqdm(range(len(self.val_ds)), desc="loading val"):
                _, preds, std, a, b, l = self.val_ds[i]
                concepts.append(preds)
                stds.append(std)
                labels.append(l)
                #print(self.val_ds[i])
                
        else:
            raise NotImplementedError()
        
        concepts = torch.stack(concepts, dim=0).float()
        stds = torch.stack(stds, dim=0)
        mask = (stds != -1).bool()
        print(stds.shape)
        print(mask.shape)
        #concepts[mask] = torch.exp(-stds[mask])*concepts[mask] + 0.5*(1-torch.exp(-stds[mask]))
        labels = torch.stack(labels, dim=0)
        print(concepts.shape, labels.shape)
        return concepts, labels
        
class _Model(torch.nn.Module):
    def __init__(self, args): #backbone_name, W_c, W_g, b_g, proj_mean, proj_std, device="cuda"):
        super().__init__()
        #args.fc_layers = []
        self.final = torch.nn.Linear(in_features = args.num_c, out_features=args.num_classes).to(args.device)
        self.args = args
    
    def forward(self, probs):
        #print(probs.shape)
        preds = self.final(probs)
        #print(probs[0:3])
        logits = torch.logit(probs, eps=1e-8)
        #print(logits[0:3])
        # The concepts are now the probs since we are approximating a GP
        out_dict = {'unnormalized_concepts':logits, 'concepts':logits, 'preds':preds, 'concept_probs':probs}
        return out_dict

    def load(self):
        # Load the final layer
        W_g = torch.load(os.path.join(self.args.load_dir, "W_g.pt"), map_location=self.args.device, weights_only=True)
        b_g = torch.load(os.path.join(self.args.load_dir, "b_g.pt"), map_location=self.args.device, weights_only=True)
        self.final.load_state_dict({"weight":W_g, "bias":b_g})
        return 
    
    def get_loss(self, args):
        return NotImplementedError('No loss implemented')
        
    def start_optim(self, args):
        return NotImplementedError('No loss implemented')
        self.opt = torch.optim.Adam(self.parameters(), args.lr)


class ARGO(BaseModel):
    def __init__(self, args):
        super().__init__(self, args)
        # Update the load_dir based on the model
        self.model = _Model(args)
        self.args = self.model.args
        self.backbone = ArgoConceptExtractor()
        
    def train(self, loader):
        pass
    
    def run(self, split = 'test'):
      logger.debug(f"Running model on {split} split.")
      self.check_integrity()
      data = GenericDataset(ds_name = self.args.dataset.split("_")[0], split = split)
      concept_ground_truth = []
      for i in range(len(data)):
          concept_ground_truth.append(data[i][1])
      concept_ground_truth = torch.stack(concept_ground_truth, dim=0).long()
      
      self.model.args.transform = str(self.get_transform(split=split))
      if split == 'test':
          (concept_annotations, gt_labels) = torch.load(os.path.join(self.args.load_dir, "test.pt"), map_location=self.args.device, weights_only=True)
      elif split == 'val':      
          (concept_annotations, gt_labels) = torch.load(os.path.join(self.args.load_dir, "val.pt"), map_location=self.args.device, weights_only=True)
      elif split == 'train':      
          (concept_annotations, gt_labels) = torch.load(os.path.join(self.args.load_dir, "train.pt"), map_location=self.args.device, weights_only=True)
      
      
      #self.backbone.load(train_x, train_y, val_x, val_y, test_x, test_y)
      self.model.load() # Load the model weights
      device = self.args.device
      
      concepts = []
      labels = []
      preds = []
      acc_mean = 0
      debug_i = 0
      n = 0
      
      temp_data = []
      for i in range(concept_annotations.shape[0]):
          temp_data.append((concept_annotations[i], gt_labels[i]))
        
      loader = torch.utils.data.DataLoader(temp_data, batch_size = self.args.batch_size, shuffle = False)
      
      for concepts_ann, targets in tqdm(loader, desc = f"Running {split}"):
        concepts_ann = concepts_ann.to(device)
        targets = targets.to(device)

        # forward pass
        with torch.no_grad():
            out_dict = self.model(concepts_ann.float())
            logits = out_dict['preds'].float()
            c_repres = out_dict['concepts']
            
            concepts.append(torch.logit(concepts_ann, eps=1e-6))
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
      
      
      concepts = torch.cat(concepts, dim=0).cpu()
      labels = torch.cat(labels, dim=0).cpu()
      preds = torch.cat(preds, dim=0).cpu()
      print(concept_ground_truth[0:3,-4:])
      print(concepts[0:3,-4:])
      
      out_dict = {
        "concepts_gt": concept_ground_truth,
        "concepts_pred": concepts,
        "labels_gt": labels,
        "labels_pred": preds,
        "accuracy": acc_mean / len(loader.dataset)
      }
      return out_dict
    #@staticmethod
    #def get_transform(split):
    #    logger.debug(f"Using argo method get_transform for {split}")
    #    return transforms.Compose([
    #        transforms.ToTensor(),
    #        transforms.Resize((224,224)),
    #    ])
    
    def get_loader(self, split):
        dataset_name = self.args.dataset
        dataset_base = dataset_name.split("_")[0]
        transform = self.get_transform(split=split)
        gt_data = GenericDataset(dataset_base, split = split, transform = transform)
        return DataLoader(gt_data, batch_size=self.args.batch_size, shuffle=False)
    
    def get_subset_loader(self, split, val_split_count = 1000):
        if split not in ['val']:
            raise ValueError()
        
        dataset_name = self.args.dataset
        transform = self.get_transform(split=split)
        gt_data = GenericDataset(dataset_name, split = split, transform = transform)
        self.subset_indices = random.sample(range(len(gt_data)), val_split_count)
        gt_data_subset = torch.utils.data.Subset(gt_data, self.subset_indices)
        self.val_data = gt_data_subset
        logger.warning(f"Using subset val data with {val_split_count} samples")
        return DataLoader(gt_data_subset, batch_size=self.args.batch_size, shuffle=False)
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

