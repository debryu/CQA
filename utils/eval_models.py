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
        X = conc_pred[:,i].numpy().reshape(-1,1)  # sklearn requires 2d input 
        y = conc_gt[:,i].numpy()
        LR = LogisticRegression(class_weight='balanced')
        LR.fit(X,y)
        w = LR.coef_[0][0]  # Slope
        b = LR.intercept_[0] # Intercept
        W.append(w)
        B.append(b)
        zeros = np.count_nonzero(LR.predict(X) == 0)
        print(zeros, len(y))
    
    return torch.tensor(W),torch.tensor(B)



