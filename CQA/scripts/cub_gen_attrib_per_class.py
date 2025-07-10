from CQA.datasets import GenericDataset
import torch
from collections import defaultdict
from tqdm import tqdm
from CQA.config import DATASETS_FOLDER_PATHS
import os 
from matplotlib import pyplot as plt
from torchvision import transforms
import json

samples = defaultdict(set)

t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
        ])

data = GenericDataset('cub',split='train')
for i in tqdm(range(len(data))):
    _, c, l = data[i]
    c = torch.tensor(c)
    #if l == 8:
    #    print(i)
    #    print(c)
        #img.save(f"./test_{i}.png")
    indices = torch.where(c == 1)[0]
    samples[l.item()].update(indices.tolist())

data = GenericDataset('cub',split='val')
for i in tqdm(range(len(data))):
    _, c, l = data[i]
    c = torch.tensor(c)
    indices = torch.where(c == 1)[0]
    samples[l.item()].update(indices.tolist())

print(samples)
with open("./data/cub/CUB_200_2011/classes.txt") as f:
    classes = f.read().split("\n")

classes = [c.replace("_"," ") for c in classes]

with open("./data/concepts/cub/cub_improved_concepts.txt") as f:
    concepts = f.read().split("\n")

# Reformat the classes
new_classes = []
for clas in classes:
    clas = clas.split(".")[-1]
    if clas != '':
        new_classes.append(clas)
#print(new_classes)
print(new_classes[8])
# Save resulting concepts
with open("./data/concepts/cub/classes.txt", 'w') as f:
    f.write(new_classes[0])
    for c in new_classes[1:]:
        f.write('\n'+c)

attributes_per_class = {}
for i,clas in enumerate(new_classes):
    active_concepts = samples[i]
    concept_list = []
    for idx in active_concepts:
        concept_list.append(concepts[idx])
    attributes_per_class[new_classes[i]] = concept_list

#key = list(attributes_per_class.keys())[8]
#print(key)
#print(attributes_per_class[key])
#print(samples)
with open("./data/concepts/cub/cub_improved_per_class.json","w") as f:
    json.dump(attributes_per_class, f, indent = 2)