import os
import ast
import math

MODELS = "./ICMLmodels"

used_metrics = ['label_f1', 'avg_macro_pr_auc', 'leakage',  'disentanglement']
experiments = {
                'celeba': {'labo': [],
                           'lfcbm': [],
                           'resnetcbm': [],
                           'vlgcbm':[]}
                        ,
                'shapes3d': {   'labo': [],
                                'lfcbm': [],
                                'resnetcbm': [],
                                'vlgcbm':[]}
                        ,
                'dermamnist': { 'labo': [],
                                'lfcbm': [],
                                'resnetcbm': [],
                                'vlgcbm':[]}
                        ,
                #'cub': {    'labo': [],
                #           'lfcbm': [],
                #           'resnetcbm': [],
                #           'vlgcbm':[]}
                #        ,
                }

folders = os.listdir(MODELS)
for folder in folders:
    model = folder.split("_")[0]
    dataset = folder.split("_")[1]
 

    with open(os.path.join(MODELS, folder, "metrics.txt"), "r") as f:
        text = f.read()

    metrics = ast.literal_eval(text)
    #print(metrics)
    experiments[dataset][model].append(metrics)
    
    
    import numpy as np

summary = {}  # will hold aggregated results
for dataset, models in experiments.items():
    summary[dataset] = {}
    for model, runs in models.items():
        print(model, dataset)
        print(runs)
        # Collect all metric names from the first run
        metrics = runs[0].keys()
        summary[dataset][model] = {}

        for metric in used_metrics:
            # gather values for this metric across all runs
            values = [run[metric] for run in runs]
            summary[dataset][model][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "num_runs": len(values)
            }

# print the aggregated results
import pprint
pprint.pprint(summary)


latex = '&\n'
for dataset in ['shapes3d']:
    for model in ['LABO','LFCBM','VLGCBM']:
        latex += f"\\{model}\n"
        
        for metric in summary[dataset][model.lower()]:
            std = summary[dataset][model.lower()][metric]['std']
            if std < 0.01:
                std = 0.01
            latex += f"& ${summary[dataset][model.lower()][metric]['mean']:.2f} \\pm {std:.2f} $     % {metric.upper()}\n"
        latex += "\\\\\n"
        latex += "&\n"
        
print(latex)