import os, json
import torch
folder = "/mnt/cimec-storage6/users/nicola.debole/home/CQA/saved_models/argus_celeba_2025_10_17_22_02/temp"

def compute_lkg(order, delta):
    size = len(order)
    values = torch.zeros(size)
    for i in range(size):
        # Take the {index} concept from the list
        index = order[i]
        
        # Add the increment observed in concept {index} found at position {i} in the list
        values[index] += delta[i]
    values /= size
    print(values)


files = os.listdir(folder)
for f in files:
    if f.endswith(".json"):
        with open(os.path.join(folder,f), "r") as f:
            data = json.load(f)
        order = data['order']
        delta = data['delta']
        compute_lkg(order,delta)