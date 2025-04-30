import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import torch
from tqdm import tqdm

def train_LR_on_concepts(conc_pred:torch.tensor,conc_gt:torch.tensor):
    n_concepts = conc_pred.shape[1]
    W = []
    B = []
    for i in tqdm(range(n_concepts), desc="Fitting Logistic Regression"):
        X = conc_pred[:,i].numpy().reshape(-1,1)# sklearn requires 2d input 
        
        y = conc_gt[:,i]
        LR = LogisticRegression(C=1,class_weight='balanced')
        LR.fit(X,y)
        w = LR.coef_[0][0]  # Slope
        b = LR.intercept_[0] # Intercept
        
        W.append(w)
        B.append(b)
        zeros = np.count_nonzero(LR.predict(X) == 0)
        print(zeros, len(y))
    
    return torch.tensor(W),torch.tensor(B)


class MultinomialLogisticRegression(torch.nn.Module): 
	def __init__(self, input_size, num_classes): 
		super(LogisticRegression, self).__init__() 
		self.linear = torch.nn.Linear(input_size, num_classes) 

	def forward(self, x): 
		out = self.linear(x) 
		out = torch.nn.functional.softmax(out, dim=1) 
		return out 



def train_LR_on_concepts_shapes3d(conc_pred:torch.tensor,conc_gt:torch.tensor):
    n_concepts = conc_pred.shape[1]
    W = []
    B = []
    concept_groups = [10, 10, 10, 8, 4] 
    start = 0
    for size in tqdm(concept_groups, desc="Fitting Logistic Regression"):
        X = conc_pred[:,start:start + size].numpy() # sklearn requires 2d input 
        print(X)
        print(X.shape)
        y = conc_gt[:,start:start + size]
        y = torch.argmax(y, dim=1)   
        start += size
        LR = LogisticRegression(class_weight='balanced', multi_class='multinomial')
        LR.fit(X,y)
        print(LR.score(X,y))
        w = LR.coef_[0][0]  # Slope
        b = LR.intercept_[0] # Intercept
        for i in range(size):
            W.append(w)
            B.append(b)
        zeros = np.count_nonzero(LR.predict(X) == 0)
        print(zeros, len(y))
    
    return torch.tensor(W),torch.tensor(B)

def train_LR_global(conc_pred:torch.tensor,conc_gt:torch.tensor):
    n_concepts = conc_pred.shape[1]
    W = []
    B = []

    X = conc_pred.numpy().reshape(-1).reshape(-1,1)  # sklearn requires 2d input 
    y = conc_gt.numpy().reshape(-1)
    print(X.shape)
    print(y.shape)
    LR = LogisticRegression(class_weight='balanced')
    LR.fit(X,y)
    w = LR.coef_[0][0]  # Slope
    b = LR.intercept_[0] # Intercept
    W.append(w)
    B.append(b)
    zeros = np.count_nonzero(LR.predict(X) == 0)
    print(zeros, len(y))
    
    return torch.tensor(W),torch.tensor(B)    



