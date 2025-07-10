import torch

class Estimator(torch.nn.Module):
    def __init__(self, Y_dim, evidence_dim, n_layers = 3, hidden_dim = 1000, linear = False):
        '''
            Train a model to estimate the probability of Y given the evidence P(Y|Evidence)
        '''
        super().__init__()
        self.hidden_fc = torch.nn.ModuleList()
        if n_layers <= 1:
            raise ValueError('Number of layers must be greater than 1. If you want to use linear layer, use linear=True instead.')
        if linear:
            self.hidden_fc.append(torch.nn.Linear(evidence_dim, Y_dim))
        else:
            self.hidden_fc.append(torch.nn.Linear(evidence_dim, hidden_dim))
            for i in range(n_layers):
                self.hidden_fc.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.hidden_fc.append(torch.nn.Linear(hidden_dim, Y_dim))
        self.activation = torch.nn.GELU()
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = x.float()
        for i in range(len(self.hidden_fc)):
            x = self.hidden_fc[i](x)
            if i != len(self.hidden_fc) - 1:
                x = self.activation(x)
        return x