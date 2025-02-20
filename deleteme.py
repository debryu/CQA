from datasets import get_dataset
from torch.utils.data import Subset
import torch
import pickle 
from datasets.dataset_classes import SKINCON_Original, CHESTMINST_Dataset
from tqdm import tqdm
from datasets import GenericDataset

data = GenericDataset('celeba', split = 'val')

sample = data[0]
c_w = data.get_concept_weights(0)
c_l = data.get_label_weights()
print(c_w)
print(c_l)
print(data.get_pos_weights())
asd
data = CHESTMINST_Dataset()
for sample in tqdm(data): 
    print(sample)
    print(sample[1].shape)
    asd


data = SKINCON_Original()
t_l = {}
n_l = {}
l = {}
i = 0
for sample in tqdm(data):
    img, nine,three,lab = sample
    l[lab] = l.get(lab,0) + 1
    #print(nine, three, lab)
    t_l[three] = t_l.get(three,0) + 1
    n_l[nine] = n_l.get(nine,0) + 1 
    i += 1


print(l)
conc = []
for key in l:
    if l[key] > 200:
        conc.append(key)
print(len(conc))
# Save concepts as a txt file
with open('filtered_concepts.txt', 'w') as f:
    for item in conc:
        f.write("%s\n" % item)
asd
a = pickle.load(open("./data/llava-phi3_annotations/shapes3d/val/query_1.pkl",'rb'))
print(a.shape)
sd

cel = get_dataset('celeba', split = 'test')
print(len(cel))
asd
id2 = range(25000,len(cel))
id1 = range(25000)

sample1 = Subset(cel, id1)
sample2 = Subset(cel, id2)

counts1 = torch.zeros(sample1[0][1].shape[0])
counts2 = torch.zeros(sample1[0][1].shape[0])
for img,c,l in sample1:
    counts1 += c

for img,c,l in sample2:
    counts2 += c

counts1 /= len(sample1)
counts2 /= len(sample2)

print(counts1)
print(counts2)