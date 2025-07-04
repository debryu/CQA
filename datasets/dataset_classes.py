import torchvision
from medmnist import ChestMNIST
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
from CQA.datasets.utils import create_dataset
from CQA.utils.utils import download_image
import traceback
import pickle

# TODO: make the mini version check if the dimension is bigger than the actual dataset
# Also could make those customizable from the terminal

# TODO: Remove celeba self.classes
# TODO: Remove self.has_concepts deprecated
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
                self.data = torchvision.datasets.CIFAR10(root=root,train=True,transform=transform,target_transform=target_transform,download=download)
                self.data = Subset(self.data,range(int(len(self.data)*train_val_split)))
                
            elif split == 'val' or split == 'valid':
                self.data = torchvision.datasets.CIFAR10(root=root,train=True,transform=transform,target_transform=target_transform,download=download)
                self.data = Subset(self.data,range(int(len(self.data)*train_val_split),len(self.data)))
            else:
                raise ValueError(f"Split {split} not recognized")
        else:
            self.data = torchvision.datasets.CIFAR10(root=root,train=False,transform=transform,target_transform=target_transform,download=download)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        x, y = self.data[index]
        return x, -1, y


class CelebAOriginal(torchvision.datasets.CelebA):
    name = "celeba_original"
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
    
class CelebA(Subset):
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
        train_subset_indices = [25000,50000],
        val_subset_indices = [0,5000],
        test_subset_indices = [0,-1],
    ) -> None:
        '''
        concepts: list of concepts to use, by choosing the indexes of the celeba attributes
        label: the index of the attribute to use as label
        '''
        assert type(label) == int # label must be an integer
        self.train_subset_indexes = train_subset_indices
        self.val_subset_indexes = val_subset_indices
        self.test_subset_indexes = test_subset_indices
        self.has_concepts = True
        if split == 'train':
            self.data = CelebAOriginal(root=root,split=split,target_type=target_type,transform=transform,target_transform=target_transform,download=download,concepts=concepts,label=label)
            self.classes = self.data.classes
            if train_subset_indices[1] == -1:
                train_subset_indices[1] = len(self.data)
            self.subset_indices = range(train_subset_indices[0],train_subset_indices[1])
            super().__init__(self.data,self.subset_indices)
        elif split == 'val' or split == 'valid':
            self.data = CelebAOriginal(root=root,split=split,target_type=target_type,transform=transform,target_transform=target_transform,download=download,concepts=concepts,label=label)
            self.classes = self.data.classes
            if val_subset_indices[1] == -1:
                val_subset_indices[1] = len(self.data)
            self.subset_indices = range(val_subset_indices[0],val_subset_indices[1])
            #self.subset_indices = range(0,len(self.data))
            super().__init__(self.data,self.subset_indices)
        elif split == 'test':
            self.data = CelebAOriginal(root=root,split=split,target_type=target_type,transform=transform,target_transform=target_transform,download=download,concepts=concepts,label=label)
            self.classes = self.data.classes
            if test_subset_indices[1] == -1:
                test_subset_indices[1] = len(self.data)
            #print(len(self.data))
            self.subset_indices = range(test_subset_indices[0],test_subset_indices[1])
            #self.subset_indices = range(0,len(self.data))
            super().__init__(self.data,self.subset_indices)

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
        train_subset_indices = [0,1000],
        val_subset_indices = [0,1000],
        test_subset_indices = [0,1000],
    ) -> None:
        '''
        concepts: list of concepts to use, by choosing the indexes of the celeba attributes
        label: the index of the attribute to use as label
        '''
        assert type(label) == int # label must be an integer
        self.has_concepts = True
        if split == 'train':
            self.data = CelebAOriginal(root=root,split=split,target_type=target_type,transform=transform,target_transform=target_transform,download=download,concepts=concepts,label=label)
            self.classes = self.data.classes
            if train_subset_indices[1] == -1:
                train_subset_indices[1] = len(self.data)
            self.subset_indices = range(train_subset_indices[0],train_subset_indices[1])
            super().__init__(self.data,self.subset_indices)
        elif split == 'val' or split == 'valid':
            self.data = CelebAOriginal(root=root,split=split,target_type=target_type,transform=transform,target_transform=target_transform,download=download,concepts=concepts,label=label)
            self.classes = self.data.classes
            if val_subset_indices[1] == -1:
                val_subset_indices[1] = len(self.data)
            self.subset_indices = range(val_subset_indices[0],val_subset_indices[1])
            #self.subset_indices = range(0,len(self.data))
            super().__init__(self.data,self.subset_indices)
        elif split == 'test':
            self.data = CelebAOriginal(root=root,split=split,target_type=target_type,transform=transform,target_transform=target_transform,download=download,concepts=concepts,label=label)
            self.classes = self.data.classes
            if test_subset_indices[1] == -1:
                test_subset_indices[1] = len(self.data)
            self.subset_indices = range(train_subset_indices[0],train_subset_indices[1])
            #self.subset_indices = range(0,len(self.data))
            super().__init__(self.data,self.subset_indices)
        
      
      
class SHAPES3DOriginal(torch.utils.data.Dataset):
    name = "shapes3d_original"
    def __init__(self, root='./data/shapes3d', split='train', transform = None, args=None):
        self.base_path = os.path.join(root, '3dshapes.h5')
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
        # Convert the image to PIL img
        image = Image.fromarray(image)
        concepts = self.concepts[idx]
        labels = self.labels[idx]
        if self.transform is not None:
            return self.transform(image), torch.tensor(concepts), torch.tensor(labels)
        else: 
            return image, concepts, int(labels)

    def __len__(self):
        return len(self.images)      
      
class SHAPES3D_Custom(Subset):
    name = "shapes3d"
    def __init__(self, root='./data/shapes3d', split='train', transform = None, args=None):
        # Check if a custom dataset partition is available
        if split == 'train':
            indexes = range(48000)
        elif split == 'val' or split == 'valid':
            indexes = range(5000)
        elif split == 'test':
            indexes = range(30000)
        else:
            raise NotImplementedError
        super().__init__(SHAPES3DOriginal(root=root,split=split,transform=transform,args=args),indexes)

class SHAPES3DMini(Subset):
    name = "shapes3d_mini"
    def __init__(self, root='./data/shapes3d', split='train', transform = None, args=None, subset_indices = [0,1000]):
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
    def __init__(self, root, split, transform=None, download=False):
        self.root = root
        
        self.pkl_urls = ['https://worksheets.codalab.org/rest/bundles/0x5b9d528d2101418b87212db92fea6683/contents/blob/class_attr_data_10/train.pkl', 
                    'https://worksheets.codalab.org/rest/bundles/0x5b9d528d2101418b87212db92fea6683/contents/blob/class_attr_data_10/test.pkl',
                    'https://worksheets.codalab.org/rest/bundles/0x5b9d528d2101418b87212db92fea6683/contents/blob/class_attr_data_10/val.pkl']
        self.cub_url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
        self.filename = 'CUB_200_2011.tgz'
        self.tgz_md5 = '97eceeb196236b17998738112f37df78'
        self.download = download
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
        #print(os.path.join(self.root,'CUB_200_2011.tgz'))
        if os.path.exists(os.path.join(self.root,'CUB_200_2011.tgz')):
            logger.debug('Files already downloaded and extracted.')
        else:
            if self.download:
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


class SKINCON_Original(Dataset):
    def __init__(self, root="./data/skincon", transform=None):
        os.makedirs(root, exist_ok=True)
        self.root = root
        self.transform = transform
        self.urls = {
                        'fitzpatrick17k.csv':'https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/refs/heads/main/fitzpatrick17k.csv',
                        'annotations_fitzpatrick17k.csv':'https://skincon-dataset.github.io/files/annotations_fitzpatrick17k.csv'
                    }
        self.files = [  'fitzpatrick17k.csv',
                        'fitzpatrick17k_images',
                        'annotations_fitzpatrick17k.csv',
                        'filtered_concepts.txt'
                     ]
        if not self._check_integrity():
            drop_indices = self._download()

        try:
            self.data = pd.read_csv(os.path.join(self.root,'fitzpatrick17k_filtered.csv'))
        except:
            self.data = pd.read_csv(os.path.join(self.root,'fitzpatrick17k.csv'))
            for entry in tqdm(self.data.itertuples(index=True), desc="Downloading SkinCon images"):
                #logger.debug(entry.url)
                if pd.isna(entry.url) or str(entry.url).strip() == '':
                    logger.warning(f"Entry {entry.Index} has no url. Dropping entry.")
                    drop_indices.append(entry.Index)
                    logger.debug(f"Dropped: {drop_indices}")
            ## Drop all collected indices at once
            self.data.drop(index=drop_indices, inplace=True)
            # Save the cleaned data
            self.data.to_csv(os.path.join(self.root,'fitzpatrick17k_filtered.csv'), index=False)

        # Read concepts from file filtered_concepts.txt
        with open(os.path.join(self.root,'filtered_concepts.txt')) as f:
            self.concepts = f.read().split("\n")
        for entry in tqdm(self.data.itertuples(index=True), desc="Filtering"):
            drop_indices = []
            if entry.label not in self.concepts:
                logger.warning(f"Entry {entry.Index} dropped because of low frequency concept {entry.label}")
                drop_indices.append(entry.Index)
                logger.debug(f"Dropped: {drop_indices}")
            ## Drop all collected indices at once
            self.data.drop(index=drop_indices, inplace=True)
            # Save the cleaned data
            self.data.to_csv(os.path.join(self.root,'fitzpatrick17k_filtered.csv'), index=False)
        self.annotations = pd.read_csv(os.path.join(self.root,'annotations_fitzpatrick17k.csv'))
        
        self.classes = {'non-neoplastic':0, 'benign':1, 'malignant':2}
        self.concepts = {concept:i for i,concept in enumerate(self.concepts)}
        #for entry in self.annotations.itertuples(index=True):
        #    print(entry)
        #    id = entry.ImageID
        #    if os.path.exists(os.path.join(self.root,'fitzpatrick17k_images',id)):
        #        print("Image exists")
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        entry = self.data.iloc[idx]
        image = Image.open(os.path.join(self.root,'fitzpatrick17k_images',f"{entry['md5hash']}.jpg")).convert('RGB')
        nine_part_label = entry['nine_partition_label']
        three_part_label = entry['three_partition_label']
        label = entry['label']
        return image, self.concepts[label], self.classes[three_part_label]
    
    def _download(self):
        # httpwwwdermaamincomsiteimagesclinicalpicLLichensimplexchronicusLichensimplexchronicus30jpg.jpg
        # http://www.dermaamin.com/site/images/clinicalpic/LLichensimplexchronicus/Lichensimplexchronicus30jpg.jpg
        for file in self.urls.keys():
            download_url(self.urls[file], self.root, file)
        self.data = pd.read_csv(os.path.join(self.root,'fitzpatrick17k.csv'))
        images_path = os.path.join(self.root,'fitzpatrick17k_images')
        os.makedirs(os.path.join(images_path), exist_ok=True)
        failed_downloads = []
        for entry in tqdm(self.data.itertuples(index=True), desc="Downloading SkinCon images"):
            url = entry.url
            name = entry.md5hash
            file_path = os.path.join(images_path,f"{name}.jpg")
            if not os.path.exists(file_path):
                logger.debug(f"Image not found. Downloading {url} to {file_path}")
                downloaded = download_image(url, file_path)
                if not downloaded:
                    failed_downloads.append(entry.Index)
        return failed_downloads

    def _check_integrity(self):
        for f in self.files:
            if not os.path.exists(os.path.join(self.root,f)):
                return False
        return True
    

class CHESTMINST_Dataset(Dataset):
    '''
    0 Atelectasis;
    1 Cardiomegaly; 
    2 Effusion; 
    3 Infiltration; 
    4 Mass; 
    5 Nodule; 
    6 Pneumonia; 
    7 Pneumothorax; 
    8 Consolidation;
    9 Edema; 
    10 Emphysema; 
    11 Fibrosis; 
    12 Pleural_Thickening; 
    13 Hernia;
    '''
    name = 'chestmnist'
    def __init__(self, root='./data/chestmnist', split='train', transform=None, size=224,
                 train_subset_indices = [0,-1],
                 val_subset_indices = [0,-1],
                 test_subset_indices = [0,-1],):
        os.makedirs(root, exist_ok=True)

        # Since the ChestMNIST dataset is grayscale, we need to convert it to RGB
        # To take advantage of the pre-trained models
        if transform is not None:
            transform_with_rgb = torchvision.transforms.Compose([
                    torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
                    *transform.transforms,
                ])
        else:
            transform_with_rgb = None
            #transform_with_rgb = torchvision.transforms.Compose([
            #        torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
            #    ])

        self.data = ChestMNIST(root=root, download=True, split=split, transform=transform_with_rgb, size=size)
        if split == 'train':
            if train_subset_indices[1] == -1:
                train_subset_indices[1] = len(self.data)
            self.data = Subset(self.data, range(train_subset_indices[0],train_subset_indices[1]))
        if split == 'val':
            if val_subset_indices[1] == -1:
                val_subset_indices[1] = len(self.data)
            self.data = Subset(self.data, range(val_subset_indices[0],val_subset_indices[1]))
        if split == 'test':
            if test_subset_indices[1] == -1:
                test_subset_indices[1] = len(self.data)
            self.data = Subset(self.data, range(test_subset_indices[0],test_subset_indices[1]))
        '''
        if split == 'train':
            self.data = Subset(self.dataset, range(0, 5000))
        elif split == 'val' or split == 'valid':
            self.data = Subset(self.dataset, range(5000, 6000))
        elif split == 'test':
            self.data = Subset(self.dataset, range(6000, 7000))
        else:
            raise NotImplementedError
        '''
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, concepts = self.data[index]
        
        label = int(np.any(concepts))
        label = torch.tensor(label)
        concepts = torch.tensor(concepts)
        return img, concepts, label