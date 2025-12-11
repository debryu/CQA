import torch
import json
from loguru import logger

last_layer = './saved_models/argus_cub_2025_08_22_17_17/W_g.pt'
# Load the weights of the last linear layer
layer = torch.load(last_layer)
try:
    weight = layer['weight']
    bias = layer['bias']
except:
    weight = layer
with open("./data/concepts/cub/cub_per_class.json", "r") as f:
    cub_per_concept = json.load(f)
with open("./data/concepts/cub/classes.txt", "r") as f:
    classes = f.read().split("\n")
with open("./data/concepts/cub/cub_improved_concepts.txt", "r") as f:
    concepts = f.read().split("\n")
    
for i in range(112):
    i_class = classes[i]
    important_concepts = cub_per_concept[i_class]
    print(important_concepts)
    sorted_vals, indices = torch.sort(torch.abs(weight[i]), descending=True)
    for ii,k in enumerate(indices):
        if sorted_vals[ii] > 0:
            if weight[i][k] > 0:
                logger.info(concepts[k])
            else:
                logger.error(concepts[k])
    input('..')