from datasets import get_dataset
from torch.utils.data import DataLoader
from metrics.completeness.estimator_model import Estimator
import torch
from metrics.completeness.completeness import train_random_concepts,train_random_labels
from torchvision.transforms import ToTensor

ds = get_dataset('celeba', split = 'train', transform = ToTensor())
dl = DataLoader(ds, batch_size = 10, shuffle = True)

model = Estimator(2, 39)
loss = torch.nn.CrossEntropyLoss()

#train_random_labels(model, dl, loss, [0.7])
train_random_concepts(model, dl, loss, [0.1,0.9,0.1,0.9,0.1,0.1,0.9,0.1,0.9,0.1,0.1,0.9,0.1,0.9,0.1,0.1,0.9,0.1,0.9,0.1,0.1,0.9,0.1,0.9,0.1,0.1,0.9,0.1,0.9,0.1,0.1,0.9,0.1,0.9,0.1,0.1,0.9,0.1,0.9])