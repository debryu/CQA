import torch

class FinalLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(in_features, out_features))
        self.softmax = torch.nn.functional.softmax(self.weight, dim=1)
        
    def forward(self, x):
        x = self.weight @ x
        x = self.softmax(x)
        return x