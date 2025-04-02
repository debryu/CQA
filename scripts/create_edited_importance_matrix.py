import os
import pickle
from loguru import logger
from matplotlib import pyplot as plt
dataset = 'celeba'
model = 'oracle'
folder = f'./ordered_models/{dataset}/{model}/{model}_{dataset}_2025_02_24_01_21'
import numpy as np
import argparse
import matplotlib

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


CQA = open_CQA(folder)
print(CQA)
importance_matrix = CQA.dci['importance_matrix']
import seaborn as sns

#y_labels = [c for c in concepts]
plt.figure(figsize=importance_matrix.shape)
ax = sns.heatmap(importance_matrix, annot=False, 
            fmt=".2f", 
            cmap="hot", 
            linewidths=0.5, 
            square=True, 
            #xticklabels=GROUND_TRUTH_CONCEPTS[dataset_name],
            #yticklabels=y_labels,
            #cbar_kws={'shrink': 0.5, 'labelsize': 10}
            cbar=False)
plt.tight_layout()
#plt.xticks(rotation=50, ha='right', va='top', )
# Remove x and y labels
ax.set_xticklabels([])
ax.set_yticklabels([])

plt.title("", fontsize=80, pad=20)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.savefig(f"./scripts/{model}_{dataset}_IM.png")
        
asd
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
            leak = CQA.metrics['leakage']
            leakage[model].append(leak)
        except:
            logger.warning(f"Missing {m}")
        
        
for model in leakage.keys():
    res = np.array(leakage[model])
    leakage_metric[model] = (np.mean(res),np.std(res))
        #except:
        #    logger.warning(f"Missing {os.path.join(folder,f,m)}")
        
    
print(leakage_metric)