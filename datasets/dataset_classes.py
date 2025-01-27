from torchvision.datasets import CelebA, CIFAR10
from torch.utils.data import Subset, Dataset
import torch
from tqdm import tqdm
import os
import numpy as np
from datasets.utils import create_dataset

# TODO: make the mini version check if the dimension is bigger than the actual dataset
# Also could make those customizable from the terminal

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
        subset_indices = [0,1000]
    ) -> None:
      '''
        concepts: list of concepts to use, by choosing the indexes of the celeba attributes
        label: the index of the attribute to use as label
      '''
      assert type(label) == int # label must be an integer
      self.has_concepts = True

      if split == 'val':
        split = 'valid'
      
      self.data = CelebACustom(root=root,split=split,target_type=target_type,transform=transform,target_transform=target_transform,download=download,concepts=concepts,label=label)
      self.classes = self.data.classes
      self.subset_indices = range(subset_indices[0],subset_indices[1])
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

import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url

class Cub2011(Dataset):
    name = "cub"
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, split='train', transform=None, download=True, args=None):
        self.root = root
        self.has_concepts = True
        self.transform = transform
        self.loader = default_loader
        self.split = split
        self.args = args
        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.split == 'train':
            self.data = self.data[self.data.is_training_img == 1]
        else:
            test_valid_data = self.data[self.data.is_training_img == 0]
            tv_len = len(test_valid_data)
            if 'val_test_ratio' in self.args:
                val_test_ratio = self.args.val_test_ratio
            else:
                val_test_ratio = 0.2
            val_len = int(tv_len * val_test_ratio)
            test_len = tv_len - val_len
            if self.split == 'test':  
              self.data = test_valid_data[:test_len]
            else:
              self.data = test_valid_data[test_len:]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target