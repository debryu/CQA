import torch
from tqdm import tqdm
#from sklearn.metrics import classification_report
import os
from torch.utils.data import DataLoader
from metrics.completeness.estimator_model import Estimator
from datasets import get_dataset
import pickle

def train_random_labels(model, dl, loss, prob_1, device = 'cuda', lr = 0.001):
    model.train()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_losses = []
    prob_1 = torch.tensor(prob_1).to(device)
    print('Training random predictor')
    for i, (_, concepts,_) in enumerate(tqdm(dl)):
        # Generate samples from bernoulli distribution with probability prob_1
        t = torch.ones(concepts.shape[0], 1).to(device)
        t = t*prob_1
        labels = torch.bernoulli(t)
        # Convert to one hot encoding
        labels = labels.view(-1).long()
        labels = torch.nn.functional.one_hot(labels, num_classes=2)
        print(labels)
        labels = labels.to(device, torch.float)
        #print(labels)
        #print(labels.shape)
        concepts = concepts.to(device)
        preds = model(concepts)
        
        loss_val = torch.mean(loss(preds, labels))
        train_losses.append(loss_val.item())
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        
    return sum(train_losses)/len(train_losses)

def train_random_concepts(model, dl, loss, prob_1, device = 'cuda', lr = 0.001):
    model.train()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_losses = []
    prob_1 = torch.tensor(prob_1).to(device)
    print('Training random predictor')
    for i, (_, concepts,labels) in enumerate(tqdm(dl)):
        # Generate samples from bernoulli distribution with probability prob_1
        t = torch.ones(concepts.shape[0],concepts.shape[1]).to(device)
        t = t*prob_1
        concepts = torch.bernoulli(t)
        # Convert to one hot encoding
        print(concepts)
        concepts = concepts.to(device, torch.float)
        labels = labels.to(device)
        #print(labels)
        #print(labels.shape)
        concepts = concepts.to(device)
        preds = model(concepts)
        
        loss_val = torch.mean(loss(preds, labels))
        train_losses.append(loss_val.item())
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        
    return sum(train_losses)/len(train_losses)