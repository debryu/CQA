from torchvision.datasets import CelebA, CIFAR10
from torch.utils.data import Subset, Dataset
import torch
from tqdm import tqdm
import os
import numpy as np
from torchvision.datasets.folder import default_loader
import pandas as pd
from torchvision.datasets.utils import download_url
from loguru import logger
from PIL import Image
from datasets.utils import create_dataset
import traceback
import pickle

# TODO: make the mini version check if the dimension is bigger than the actual dataset
# Also could make those customizable from the terminal

# TODO: Remove celeba self.classes

# TODO: Common class:
#       - get_train_val_loader
#       - get_test_loader
'''-------------------------------------------------------------------------'''
'''
    TORCHVISION DATASETS
'''
'''-------------------------------------------------------------------------'''

class Cifar10Custom(Dataset):
    name = "cifar10"
    def __init__(
        self,
        root,
        split: str = "train",
        target_type = "attr",
        transform = None,
        target_transform = None,
        download: bool = True,
        train_val_split = 0.9,
    ) -> None:
        '''
        '''
        self.has_concepts = False
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        if split == 'train':
            if split == 'train':
                self.data = CIFAR10(root=root,train=True,transform=transform,target_transform=target_transform,download=download)
                self.data = Subset(self.data,range(int(len(self.data)*train_val_split)))
                
            elif split == 'val' or split == 'valid':
                self.data = CIFAR10(root=root,train=True,transform=transform,target_transform=target_transform,download=download)
                self.data = Subset(self.data,range(int(len(self.data)*train_val_split),len(self.data)))
            else:
                raise ValueError(f"Split {split} not recognized")
        else:
            self.data = CIFAR10(root=root,train=False,transform=transform,target_transform=target_transform,download=download)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        x, y = self.data[index]
        return x, -1, y


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
      self.has_concepts = True
      self.classes = ["5_o_Clock_Shadow","Arched_Eyebrows","Attractive","Bags_Under_Eyes","Bald","Bangs","Big_Lips","Big_Nose","Black_Hair","Blond_Hair",
                      "Blurry","Brown_Hair","Bushy_Eyebrows","Chubby","Double_Chin","Eyeglasses","Goatee","Gray_Hair","Heavy_Makeup","High_Cheekbone",
                      "Mouth_Slightly_Open","Mustache","Narrow_Eyes","No_Beard","Oval_Face","Pale_Skin","Pointy_Nose","Receding_Hairline","Rosy_Cheeks","Sideburns",
                      "Smiling","Straight_Hair","Wavy_Hair","Wearing_Earrings","Wearing_Hat","Wearing_Lipstick","Wearing_Necklace","Wearing_Necktie","Young"]
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
        train_subset_indices = [0,10000],
        val_subset_indices = [0,1000],
        test_subset_indices = [0,10000],
    ) -> None:
        '''
        concepts: list of concepts to use, by choosing the indexes of the celeba attributes
        label: the index of the attribute to use as label
        '''
        assert type(label) == int # label must be an integer
        self.has_concepts = True
        if split == 'val':
            split = 'valid'
        if split == 'train':
            self.data = CelebACustom(root=root,split=split,target_type=target_type,transform=transform,target_transform=target_transform,download=download,concepts=concepts,label=label)
            self.classes = self.data.classes
            self.subset_indices = range(train_subset_indices[0],train_subset_indices[1])
            super().__init__(self.data,self.subset_indices)
        elif split == 'val' or split == 'valid':
            self.data = CelebACustom(root=root,split=split,target_type=target_type,transform=transform,target_transform=target_transform,download=download,concepts=concepts,label=label)
            self.classes = self.data.classes
            self.subset_indices = range(val_subset_indices[0],val_subset_indices[1])
            #self.subset_indices = range(0,len(self.data))
            super().__init__(self.data,self.subset_indices)
        elif split == 'test':
            self.data = CelebACustom(root=root,split=split,target_type=target_type,transform=transform,target_transform=target_transform,download=download,concepts=concepts,label=label)
            self.classes = self.data.classes
            self.subset_indices = range(test_subset_indices[0],test_subset_indices[1])
            #self.subset_indices = range(0,len(self.data))
            super().__init__(self.data,self.subset_indices)
        
      
      
      
      
class SHAPES3D_Custom(torch.utils.data.Dataset):
    name = "shapes3d"
    def __init__(self, root='./data/shapes3d', split='train', transform = None, args=None):
        self.base_path = os.path.join(root, '3dshapes.h5')
        self.has_concepts = True
        # Check if the dataset is already created
        if not os.path.exists(os.path.join(root, split+'_split_imgs.npy')) or not os.path.exists(os.path.join(root, split+'_split_cl.npy')):
            create_dataset(root, args)
        self.images= np.load(os.path.join(root, split+'_split_imgs.npy'), allow_pickle=True)

        self.classes = ['not red pill', 'red pill']
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

class SHAPES3DMini(Subset):
    name = "shapes3d_mini"
    def __init__(self, root='./data/shapes3d', split='train', transform = None, args=None, subset_indices = [0,10000]):
        self.data = SHAPES3D_Custom(root=root,split=split,transform=transform,args=args)
        self.classes = self.data.classes
        self.subset_indices = range(subset_indices[0],subset_indices[1])
        super().__init__(self.data,self.subset_indices)

'''-------------------------------------------------------------------------'''
'''
    CUSTOM IMPLEMENTED DATASETS
'''
'''-------------------------------------------------------------------------'''

class CUBDataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """
    name = 'cub'
    def __init__(self, root, split, transform=None):
        self.root = root
        
        self.pkl_urls = ['https://worksheets.codalab.org/rest/bundles/0x5b9d528d2101418b87212db92fea6683/contents/blob/class_attr_data_10/train.pkl', 
                    'https://worksheets.codalab.org/rest/bundles/0x5b9d528d2101418b87212db92fea6683/contents/blob/class_attr_data_10/test.pkl',
                    'https://worksheets.codalab.org/rest/bundles/0x5b9d528d2101418b87212db92fea6683/contents/blob/class_attr_data_10/val.pkl']
        self.cub_url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
        self.filename = 'CUB_200_2011.tgz'
        self.tgz_md5 = '97eceeb196236b17998738112f37df78'
        # Create folders
        os.makedirs(self.root,exist_ok=True)
        os.makedirs(os.path.join(self.root,'class_attr_data_10'), exist_ok=True)
        self.splits = ['train.pkl','test.pkl','val.pkl']
        self.pkl_file_paths = [os.path.join(self.root,'class_attr_data_10',split) for split in self.splits]
        # Download CUB
        self._download()
        self.data = []
        self.has_concepts = True
        self.is_train = any(["train" in path for path in self.pkl_file_paths])
        if not self.is_train:
            assert any([("test" in path) or ("val" in path) for path in self.pkl_file_paths])
        
        if split == 'train':
            self.data.extend(pickle.load(open(self.pkl_file_paths[0], 'rb')))
        elif split == 'test':
            self.data.extend(pickle.load(open(self.pkl_file_paths[1], 'rb')))
        elif split == 'val':
            self.data.extend(pickle.load(open(self.pkl_file_paths[2], 'rb')))
        else:
            raise NotImplementedError
        
        self.transform = transform
        self.image_dir = os.path.join(root,'CUB_200_2011','images')

    def _download(self):
        import tarfile
        # Download CUB images
        #print(os.path.join(self.root,'CUB_200_2011.tgz'))
        if os.path.exists(os.path.join(self.root,'CUB_200_2011.tgz')):
            logger.debug('Files already downloaded and extracted.')
        else:
            logger.info('Downloading CUB...')
            download_url(self.cub_url, self.root, self.filename, self.tgz_md5)
        if not os.path.exists(os.path.join(self.root,'CUB_200_2011/')):
            logger.info("Extracting CUB_200_2011.tgz")
            with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
                tar.extractall(path=self.root)
        # Download CUB attributes
        for i,p in enumerate(self.pkl_file_paths):
            if not os.path.exists(p):
                logger.debug(f"Downloading {p}")
                download_url(self.pkl_urls[i],os.path.join(self.root,'class_attr_data_10'), self.splits[i])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data['img_path']
        idx = img_path.split('/').index('CUB_200_2011')
        img_path = '/'.join([self.image_dir] + img_path.split('/')[idx+2:])
        img = Image.open(img_path).convert('RGB')
        class_label = img_data['class_label']
        if self.transform:
            img = self.transform(img)
        attr_label = img_data['attribute_label']
            
        return img, torch.tensor(attr_label), torch.tensor(class_label)
