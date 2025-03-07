import torch.optim
from datasets import get_dataset
from torch.utils.data import Subset
import torch
import pickle 
from datasets.dataset_classes import SKINCON_Original, CHESTMINST_Dataset
from tqdm import tqdm
from datasets import GenericDataset
from torchvision import transforms
import numpy as np
t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
        ])
ds_train = GenericDataset('cub', split = 'train', transform = t)
model = torch.nn.Linear(112,200).to('cuda')
torch.nn.init.zeros_(model.weight)
torch.nn.init.zeros_(model.bias)
loader = torch.utils.data.DataLoader(ds_train, shuffle=True, batch_size=64)
loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
optmizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

best_loss_discrete = 1000000
for e in range(10):
    train_losses = []
    #print(output['collapsed_concepts'][0])
    for batch in tqdm(loader):
        _, _in, _gt = batch
        #print(_in)
        _in = 2000*(_in.to('cuda').to(torch.float) - 0.5)
        #print(_in)
        _gt = _gt.to('cuda').to(torch.long)
        #print(_in)
        #print(output['concepts_gt'])
        result = model(_in)
        #print(result)
        #print(_gt)
        loss = loss_fn(result,_gt)
        train_losses.append(loss.item())
        optmizer.zero_grad()
        loss.backward()
        optmizer.step()
    print(np.mean(train_losses))
    if np.mean(train_losses) < best_loss_discrete:
        best_loss_discrete = np.mean(train_losses)
    #logger.info(f"Epoch {e} train loss: {np.mean(train_losses)}")

weights = model.weight
print(best_loss_discrete)
print(torch.topk(weights,3)[1])

asd
dataloader = torch.utils.data.DataLoader(ds_train, batch_size=1024, shuffle=True)
lin = torch.nn.Linear(42,2).to('cuda')
loss_fn = torch.nn.CrossEntropyLoss(reduction = 'mean')
optim = torch.optim.Adam(lin.parameters(), lr = 0.01)
for e in range(200):
    t_losses = []
    for batch in dataloader:
        _,c,l = batch
        c = c.to('cuda').float()
        l = l.to('cuda').long()
        preds = lin(c)
        optim.zero_grad()
        loss = loss_fn(preds, l)
        optim.step()
        t_losses.append(loss.item())
    print(np.mean(t_losses))
asd


occurred = []
for i in range(len(data)):
    img, c, l = data[i]
    #print(l.item())
    if l.item() in occurred:
        continue
    else:
        occurred.append(l.item())
        print(l.item())

with open("./data/cub/CUB_200_2011/classes.txt") as f:
    classes = f.read().split("\n")


for o in occurred:
    print(classes[o])
print(len(classes))
asd


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