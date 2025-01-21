import os
from loguru import logger
import torch
import random
from torch.utils.data import DataLoader, TensorDataset
from models.glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from tqdm import tqdm

from utils.lfcbm_utils import cos_similarity_cubed_single, save_activations, get_save_names, get_targets_only
from datasets import get_dataset
from config import folder_naming_convention, ACTIVATIONS_PATH
from utils.llamaoracle_utils import query_llama

def train(args):
    queries = ["a bald person","a person with a beard","a person with heavy makeup"]
    ds = get_dataset('celeba_mini', subset_indices=[0,10], split='train')
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    print(query_llama(dl, queries))

    return args
