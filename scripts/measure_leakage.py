import os
import pickle
from loguru import logger
from matplotlib import pyplot as plt
dataset = 'celeba'
folder = f'./ordered_models/{dataset}'
import numpy as np
subfolders = os.listdir(folder)
shapes = ['p','s','o','^','v']

import argparse
import matplotlib
from matplotlib import rc
#rc('font', **{'family':'serif', 'serif':['Computer Modern Roman'], 
#                                'monospace': ['Computer Modern Typewriter']})
params = {'backend': 'pdf',
          'axes.labelsize': 12,
          #'text.fontsize': 12,
          'legend.fontsize': 12,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'text.usetex': False,
          'figure.figsize': (6, 4),
          'axes.unicode_minus': True}
#matplotlib.rcParams.update(params)

from core.concept_quality import open_CQA

parser = argparse.ArgumentParser(description="Evaluate models")
parser.add_argument("-folder", type=str, default=None, help="Directory containing multiple models to test")
parser.add_argument("-eval_seed", type=int, default=42, help="Seed for the evaluation")
parser.add_argument("-load_dir", type=str, default=None, help="Load directory for the model. If not provided, the path in the config will be used")
parser.add_argument("-force", action="store_true", help="Force the computation from scratch")
parser.add_argument("-all", action="store_true", help="Compute all possible metrics")
parser.add_argument("-leakage", action="store_true", help="Compute leakage metrics")
parser.add_argument("-ois", action="store_true", help="Compute ois metrics")
parser.add_argument("-wandb", action="store_true", help="Logs on wandb")
parser.add_argument("-dci", action="store_true", help="Compute DCI")
parser.add_argument("-concept_metrics", action="store_true", help="Compute metrics concept wise")
parser.add_argument("-label_metrics", action="store_true", help="Compute classification report")
parser.add_argument("-logger", type=str, default="DEBUG", help="Logging level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
#parser.add_argument("-model", type=str, help="Model to use", choices=["vlgcbm", "lfcbm","resnetcbm"])
args = parser.parse_args()

model_translation = {
    'labo':r"LaBo",
    'lfcbm':r'LF-CBM',
    'oracle':r'$Na\"ive$',
    'resnetcbm':r'CBM',
    'vlgcbm':r'VLG-CBM', 
}

def compute_leakage_from_f1s(accuracies_gt,accuracies_pred):
    increase = []
    delta = []
    old = 0
    accuracies_gt = np.array(accuracies_gt)
    accuracies_pred = np.array(accuracies_pred)
    max_f1_gt = np.max(accuracies_gt)
    k = len(accuracies_gt)
    Z = 1-np.mean(accuracies_gt)
    gap = np.sum(np.maximum(accuracies_pred-accuracies_gt,0))
    leak = gap/(k*Z)
    return leak
    for i,f1 in enumerate(accuracies_pred):
        increment = accuracies_pred[i]-accuracies_gt[i]
        #old = accuracies_pred[i]
        if increment < 0:
            increment = 0
        normalizer = max_f1_gt-accuracies_gt[i]
        
        # When the last one concept is added and normalizer is 0, set it to one to forget about it
        if normalizer == 0.0:
            normalizer = 1.0
            
        print(normalizer)
        # If there is style leakage change the leakage computation
        if accuracies_pred[i] > max_f1_gt:
            increase.append(1 + accuracies_pred[i] - max_f1_gt)
        else:
            increase.append(increment/normalizer)
        delta.append(accuracies_pred[i]-accuracies_gt[i])
    return np.mean(increase)

# Load from the models
leakage = {'resnetcbm':[],'oracle':[],'labo':[],'lfcbm':[],'vlgcbm':[]}
leakage_metric = {}
for f in subfolders:
    models = os.listdir(os.path.join(folder,f))
    for m in models:
        model = m.split("_")[0]
        #try:
            # Load metrics
        args.folder = os.path.join(folder,f,m)
        CQA = open_CQA(os.path.join(folder,f,m))
        try:
            asd # Make sure error is thrown so that the second method is used (loaded from pickle)
            leak = CQA.metrics['leakage']
            print(CQA.metrics)
            leakage[model].append(leak)
        except:
            logger.warning(f"Missing {m}")
            lkg_info = pickle.load(open(os.path.join(folder,f,m,"leakage.data"), 'rb'))
            leakage[model].append(compute_leakage_from_f1s(lkg_info['f1_gt'],lkg_info['f1_pred']))
                
        
        
for model in leakage.keys():
    res = np.array(leakage[model])
    leakage_metric[model] = (np.mean(res),np.std(res),res.shape)
        #except:
        #    logger.warning(f"Missing {os.path.join(folder,f,m)}")
        
    
print(leakage_metric)