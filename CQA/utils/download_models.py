import torch
from torchvision import models
import torch.utils.model_zoo as model_zoo
from pytorchcv.model_provider import get_model as ptcv_get_model
from CQA.utils.clip import load

clip_models = ['RN50',"ViT-B/16","ViT-L/14"]
for model in clip_models:
    load(model)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model_zoo.load_url(model_urls['resnet18'])
ptcv_get_model("resnet18_cub", pretrained=True)