from datasets import get_dataset
from torch.utils.data import Subset
import torch
import pickle 
from datasets.dataset_classes import SKINCON_Original, CHESTMINST_Dataset
from tqdm import tqdm
from datasets import GenericDataset
import numpy as np
from torchvision import transforms


class linearModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(linearModel, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.linear(x)
    

class FinalLayer():
    def __init__(self, in_features, out_features, lr, lam, alpha, device):
        self.linear = torch.nn.Linear(in_features, out_features).to(device)
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.linear.parameters(), lr = self.lr)
        self.lam = lam
        self.alpha = alpha
        

    def train(self,loader,loss_fn):
        train_losses = []
        for _, input, label in tqdm(loader):
            input = input.to('cuda').to(torch.float32)
            label = label.to('cuda')
            output = self.linear(input)
            # ELASTIC LOSS
            weight, _ = list(self.linear.parameters())
            l1 = self.lam * self.alpha * weight.norm(p=1)
            l2 = 0.5 * self.lam * (1 - self.alpha) * (weight**2).sum()
            loss = loss_fn(output, label) + l1 + l2
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            train_losses.append(loss.item())
        return np.mean(train_losses)



lam = 0.01
alpha = 0.9
t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
        ])
dataset = GenericDataset('celeba', split = 'train', transform = t)
loader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle = True)
loss_fn = torch.nn.CrossEntropyLoss()
last_layer = FinalLayer(39,2,0.001,lam,alpha,'cuda')
for epoch in range(10):
    loss = last_layer.train(loader, loss_fn)
    print(loss)
    

print(last_layer.linear.weight)
    
asd




data = GenericDataset('celeba', split = 'val')

sample = data[0]
c_w = data.get_concept_weights(0)
c_l = data.get_label_weights()
print(c_w)
print(c_l)
print(data.get_pos_weights())
print('ciao')
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