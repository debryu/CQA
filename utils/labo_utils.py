import torch
import torch.nn.functional as F
from loguru import logger

class FinalLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, asso_act = 'softmax'):
        super().__init__()
        logger.debug(f"Initializing Final layer - in: {in_features} out:{out_features}")
        #self.weight = torch.nn.Parameter(torch.randn(in_features, out_features))
        self.asso_mat = self.create_matrix(out_features,in_features)
        self.asso_act = asso_act

    def _get_weight_mat(self):
            if self.asso_act == 'relu':
                mat = F.relu(self.asso_mat)
            elif self.asso_act == 'tanh':
                mat = F.tanh(self.asso_mat) 
            elif self.asso_act == 'softmax':
                mat = F.softmax(self.asso_mat, dim=-1) 
            else:
                mat = self.asso_mat
            return mat 

    def create_matrix(self,num_classes, num_concepts):
        init_weight = torch.zeros((num_classes, num_concepts)) #init with the actual number of selected index
        torch.nn.init.kaiming_normal_(init_weight)
        asso_mat = torch.nn.Parameter(init_weight.clone())
        return asso_mat


    def forward(self, score):
            mat = self._get_weight_mat()
            sim = score @ mat.t()
            return sim
