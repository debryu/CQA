from torchvision.datasets import CelebA
from torch.utils.data import Subset
import torch
from tqdm import tqdm
import os
import numpy as np
from datasets.utils import create_dataset

class CelebACustom(CelebA):
    name = "celeba"
    def __init__(
        self,
        root,
        split: str = "train",
        target_type = "attr",
        transform = None,
        target_transform = None,
        download: bool = False,
        concepts: list = None,
        label:int = 20,
    ) -> None:
      '''
        concepts: list of concepts to use, by choosing the indexes of the celeba attributes
        label: the index of the attribute to use as label
      '''
      assert type(label) == int # label must be an integer
      if split == 'val':
        split = 'valid'
      super().__init__(root=root,split=split,target_type=target_type,transform=transform,target_transform=target_transform,download=download)
      if concepts is None:
        self.concepts = list(range(40))
      else:
        self.concepts = concepts
      self.label = label
      if self.label in self.concepts:
        self.concepts.remove(self.label)
        #print(f"Removed label {self.label} from concepts")

    def __getitem__(self, index: int):
        x, y = super().__getitem__(index)
        concepts = y[self.concepts]
        y = y[self.label]
        return x, concepts, y
    
class CelebAMini(Subset):
    name = "celeba_mini"
    def __init__(
        self,
        root,
        split: str = "train",
        target_type = "attr",
        transform = None,
        target_transform = None,
        download: bool = False,
        concepts: list = None,
        label:int = 20,
        subset_indices = [0,10000]
    ) -> None:
      '''
        concepts: list of concepts to use, by choosing the indexes of the celeba attributes
        label: the index of the attribute to use as label
      '''
      assert type(label) == int # label must be an integer
      if split == 'val':
        split = 'valid'
      
      self.data = CelebACustom(root=root,split=split,target_type=target_type,transform=transform,target_transform=target_transform,download=download,concepts=concepts,label=label)
      self.subset_indices = range(subset_indices[0],subset_indices[1])
      super().__init__(self.data,self.subset_indices)
      
class SHAPES3D_Custom(torch.utils.data.Dataset):
    name = "shapes3d"
    def __init__(self, root='./data/shapes3d', split='train', transform = None, args=None):
        self.base_path = os.path.join(root, '3dshapes.h5')
        # Check if the dataset is already created
        if not os.path.exists(os.path.join(root, split+'_split_imgs.npy')) or not os.path.exists(os.path.join(root, split+'_split_cl.npy')):
            create_dataset(root, args)
        self.images= np.load(os.path.join(root, split+'_split_imgs.npy'), allow_pickle=True)

        concepts_and_labels = np.load(os.path.join(root, split+'_split_cl.npy'), allow_pickle=True)
        self.labels = concepts_and_labels[:,-1]
        self.concepts = concepts_and_labels[:,:-1]
        self.transform = transform

        
    def __getitem__(self, idx):
        image = self.images[idx]
        concepts = self.concepts[idx]
        labels = self.labels[idx]
        return self.transform(image), concepts, labels

    def __len__(self):
        return len(self.images)
