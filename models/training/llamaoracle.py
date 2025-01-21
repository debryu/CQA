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
    ds = args.dataset.split("_")[0]
    path = CONCEPT_SETS[ds]
    with open(path, 'r') as f:
        concepts = f.read().split("\n")
    queries = []
    for c in concepts:
        queries.append(f"a person with {c}")
    print(queries)
    t = transforms.Compose([
    transforms.Lambda(lambda x: np.array(x))  # Ensure it's a NumPy array
    ])
    start = args.start_idx
    end = args.end_idx
    ds = get_dataset(args.dataset, subset_indices=[start,end], split='train', transform=t)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    retrieved_concepts = query_llama(dl, queries)
    torch.save(retrieved_concepts, os.path.join(args.save_dir,f"{args.dataset}_{start}-{end}"))
    return args

