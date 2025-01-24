import torch

a = torch.load('./data/llama_annotations/celeba_mini_0-1000_train.pth')

b = torch.load('./data/llama_annotations/celeba_mini_0-1000_val.pth')

c = torch.load('./data/llama_annotations/shapes3d_mini_0-1000_train.pth')
print(a.shape)
print(b.shape)
print(c)