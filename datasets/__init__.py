from config import DATASETS_FOLDER_PATHS
import datasets.dataset_classes as dataset_classes
import inspect
import torch
from loguru import logger
import json
import os
import numpy as np

classes = {}
# Get all classes from the module
for name, cls in inspect.getmembers(dataset_classes, inspect.isclass):
    try:
      name = cls.name
    except:
      continue
    classes[name] = cls


logger.debug(f"Available datasets: {classes}")



class GenericDataset():
    def __init__(self, ds_name, **kwargs):
        self.dataset = get_dataset(ds_name, **kwargs)
        base = ds_name.split("_")[0]
        if 'root' in kwargs:
          self.root = kwargs['root']
        else:
          self.root = DATASETS_FOLDER_PATHS[base]

        logger.debug(f"Loading dataset {ds_name} from {self.root}")
        if os.path.exists(f"{self.root}/dataset_info.json"):
          dataset_info = json.load(open(f"{self.root}/dataset_info.json"))
          self.total_samples = dataset_info["total_samples"]
          self.concept_occurrencies = torch.tensor(dataset_info["concept_frequencies"])
          self.label_occurrencies = dataset_info["label_frequencies"]
          self.n_concepts = dataset_info["n_concepts"] 
          self.concept0_weights = torch.tensor(dataset_info["concept0_weights"])
          self.concept1_weights = torch.tensor(dataset_info["concept1_weights"])
          self.label_weights = torch.tensor(dataset_info["label_weights"]) 
        else:
          self.total_samples = len(self.dataset)
          self.n_concepts = len(self.dataset[0][1])
          self.concept_occurrencies, self.label_occurrencies = self.get_occurrencies()

          self.concept0_weights,self.concept1_weights = self._compute_concept_weights_tuple()
          self.label_weights = self._compute_label_weights()

          dataset_info = {
              "total_samples": self.total_samples,
              "concept_frequencies": self.concept_occurrencies.tolist(),
              "label_frequencies": self.label_occurrencies,
              "n_concepts": self.n_concepts,
              "concept0_weights": self.concept0_weights.tolist(),
              "concept1_weights": self.concept1_weights.tolist(),
              "label_weights": self.label_weights.tolist()
          }
          with open(f"{self.root}/dataset_info.json", "w") as f:
              json.dump(dataset_info, f, indent=4)
          
    def get_concept_weights(self, index) -> torch.Tensor:
      w = torch.zeros(2)
      w[0] = self.concept0_weights[index]
      w[1] = self.concept1_weights[index]
      return w
    def get_label_weights(self) -> torch.Tensor:
      return self.label_weights
    
    def get_pos_weights(self) -> list[int]:
      p_w = []
      for i in range(self.n_concepts):
        p_w.append(((self.total_samples-self.concept_occurrencies[i])/(self.concept_occurrencies[i])).item())
      return p_w
    
    def _compute_concept_weights_tuple(self) -> tuple[torch.Tensor, torch.Tensor]:
      weights_class0 = self.total_samples/((self.total_samples-self.concept_occurrencies)*2)
      weights_class1 = self.total_samples/(self.concept_occurrencies*2)
      return (weights_class0, weights_class1)
    def _compute_label_weights(self) -> torch.Tensor:
      n_classes = len(self.label_occurrencies)
      label_occurrencies = torch.zeros(n_classes)
      for i in range(n_classes):
        label_occurrencies[i] = self.label_occurrencies[i]
      weights = self.total_samples/(label_occurrencies*n_classes)
      return weights

    def get_occurrencies(self) -> tuple[torch.Tensor,dict]:
      concept_occ = torch.zeros(self.n_concepts)
      labels = {}
      for sample in self.dataset:
        _, concepts, label = sample
        concept_occ += torch.tensor(concepts)
        label = int(torch.tensor(label).item())
        labels[label] = labels.get(label,0) + 1
      return concept_occ, labels
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
        
    

def get_dataset(ds_name,**kwargs):
  if ds_name.endswith("_oracle") and kwargs['split'] == 'test':
    ds_name = ds_name.split("_")[0]
  if ds_name not in classes:
    raise ValueError(f"Dataset {ds_name} not found in dataset_classes")
  else:
    logger.debug(f"Getting dataset {ds_name} with kwargs {kwargs}")
    base = ds_name.split("_")[0]
    if 'root' in kwargs:
      return classes[ds_name](**kwargs)
    return classes[ds_name](root = DATASETS_FOLDER_PATHS[base], **kwargs)
  '''
  if not ds_name.endswith("_mini"):
    if ds_name.endswith("temp"):
      return classes[ds_name](**kwargs)
    if ds_name not in DATASETS_FOLDER_PATHS:
      raise ValueError(f"Dataset {ds_name} not found in DATASETS_FOLDER_PATHS")
    return classes[ds_name](root = DATASETS_FOLDER_PATHS[ds_name], **kwargs)
  else:
    original_ds_name = ds_name.split("_mini")[0]
    if original_ds_name not in DATASETS_FOLDER_PATHS:
      raise ValueError(f"Dataset {original_ds_name} not found in DATASETS_FOLDER_PATHS")
    return classes[ds_name](root = DATASETS_FOLDER_PATHS[original_ds_name], **kwargs)
  '''