import os
import pandas as pd
import os
import json
from loguru import logger
import traceback
from datetime import datetime

FAST_STORAGE = os.environ["FAST"]
default_results_folder = os.path.join(FAST_STORAGE,"results","GPs","GPs")
ARGO_activations_folder = os.path.join(FAST_STORAGE,"results","annotations")
seeds = [3]
datasets = ['dermamnist']
acq_fns = ['ucbf']
kernels = ['cos']
Ks = [7*10,7*20,7*30,7*40,7*50,7*60]

def get_activations(seeds, datasets,acq_fns,kernels,Ks, results_folder = None):
    if results_folder == None:
        results_folder = default_results_folder
    experiments = []
    # Collect all experiments
    files = os.listdir(results_folder)
    for f in files:
        date = f.split("-")[0]
        # Parse components
        year, month, day, hour, minute = map(int, date.split("_"))

        # Create datetime, assume hour=0, second=0
        dt = datetime(year, month, day, hour, minute, 0)

        # Convert to Unix timestamp
        unix_time = dt.timestamp()
        run = f.split("-")[1].replace(".json","")
        seed = int(run.split("SEED=")[-1])
        model = run.split("_")[0]
        dataset = run.split("_")[1]
        acq_fn = run.split("_")[2]
        kernel = run.split("_")[3]
        K = int(run.split("_")[4])
        with open(os.path.join(results_folder,f), 'r') as jsonfile:
            results_dict = json.load(jsonfile)
        
        if model == "SVGP":
            experiments.append({
                "path": f,
                "time": dt,
                "strtime": date,
                "seed": seed,
                "model":model,
                "dataset":dataset,
                "acq_fn":acq_fn,
                "kernel":kernel,
                "K":K,
                "results": results_dict
            })

    run = []
    # find experiments that match the parameters
    for exp in experiments:
        if exp['seed'] in seeds and exp['kernel'] in kernels and exp['acq_fn'] in acq_fns and exp['dataset'] in datasets:
            run.append(exp)
        else: 
            continue
            #print(exp['seed'], exp['kernel'],exp['acq_fn'], exp['dataset'])

    acts = []
    for r in run:
        for K in Ks:
            train_path = os.path.join(ARGO_activations_folder,f"TRAIN-{r['dataset']}-{K}-{r['acq_fn']}-{r['kernel']}-{r['seed']}_{r['strtime']}.pt")
            val_path = os.path.join(ARGO_activations_folder,f"VAL-{r['dataset']}-{K}-{r['acq_fn']}-{r['kernel']}-{r['seed']}_{r['strtime']}.pt")
            results = None
            for iter in r['results']:
                if iter['training_pool_size'] == K:
                    results = iter
                    break
            
            acts.append({"train":train_path,
                         "val":val_path,
                         "dataset":r['dataset'],
                         "seed":r['seed'],
                         "K":K,
                         "results": results,
                         "acq_fn":r['acq_fn'],
                         "strtime": r['strtime'],
                         })
    return acts

#print(get_activations(seeds,datasets,acq_fns,kernels,Ks))