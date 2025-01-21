import os
from loguru import logger
import torch
import random
from torch.utils.data import DataLoader, TensorDataset
from models.glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from tqdm import tqdm
import torchvision.transforms as transforms
import numpy as np
from datasets import get_dataset
from config import folder_naming_convention, ACTIVATIONS_PATH, CONCEPT_SETS
from utils.llamaoracle_utils import query_llama


def train(args):
    # load concepts 
    path = CONCEPT_SETS[args.dataset]
    with open(path, 'r') as f:
        concepts = f.read().split("\n")
    queries = []
    for c in concepts:
        queries.append(f"a person with {c}")
    print(queries)
    t = transforms.Compose([
    transforms.Lambda(lambda x: np.array(x))  # Ensure it's a NumPy array
    ])
    ds = get_dataset('celeba_mini', subset_indices=[0,4], split='train', transform=t)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    print(query_llama(dl, queries))

    return args

