from datasets import get_dataset
from torch.utils.data import Subset
import torch
import pickle 

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