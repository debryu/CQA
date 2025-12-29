import os, itertools, subprocess, datetime
from CQA.utils.load_ARGO_activations import get_activations
import traceback
try:
    datasets = ['dermamnist']
    acq_fns = ['ucbf']
    kernels = ['cos']
    seeds = [2]
    Ks = [7*10,7*20,7*30,7*40,7*50,7*60]
    runs = get_activations(seeds, datasets, acq_fns, kernels, Ks)
    combinations = list(runs)


    # Get combination for this SLURM array task
    idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
    dataset = combinations[idx]

    time_date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    outfile = f"{os.environ['FAST']}/results/CBMs/ARGO-CBM_{time_date}_{dataset['dataset']}_{dataset['seed']}_{dataset['K']}.txt"

    # Set env vars for the experiment
    env = os.environ.copy()


    # Run the experiment
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as f:
        result = subprocess.run(["python", 
                                 "train.py", 
                                 "-m", "argo",
                                 "-pool_size", str(dataset['K']),
                                 "-d", dataset['dataset'],
                                 "-seed", str(dataset['seed']),
                                 "-argo_train", dataset['train'],
                                 "-argo_val", dataset['val']
                                 ], env=env, capture_output=True, text=True)
        f.write(result.stdout)
        if result.stderr:
            f.write("\nERRORS:\n")
            f.write(result.stderr)

except Exception as e:
    time_date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    logfile = f"{os.environ['FAST']}/results/CBMs/logs_{time_date}.txt"
    #traceback.print_exc()
    with open(logfile, "w") as f:
        f.write(str(e))