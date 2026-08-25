#!/bin/bash
#SBATCH --job-name=train_argo
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --time=23:00:00
#SBATCH --array=0-3%2            # Adjust based on #combinations and concurrency
#SBATCH --output=/leonardo_scratch/fast/IscrC_ARGO/results/logs/train-slurm-%A_%a.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_ARGO/results/logs/train-slurm-%A_%a.err

module load python
module load cuda
source /leonardo/home/userexternal/ndebole0/projects/vlm/bin/activate


# Map SLURM_ARRAY_TASK_ID to your experiment combination
python - <<'END'
import os, itertools, subprocess, datetime
from CQA.utils.load_ARGO_activations import get_activations
import traceback

results_folder = f"{os.environ['FAST']}/results/ICML/GPs/"

try:
    datasets = ['cub']
    acq_fns = ['random']
    kernels = ['cosb']
    argo_seeds = [108]
    n_concepts_run = 15
    base = 112
    
    
    combinations = []
    for dataset in datasets:
        if dataset == 'dermamnist':
            Ks = [160,240,360]
        elif dataset == 'celeba':
            Ks = [160,240,360]
        elif dataset == 'shapes3d':
            Ks = [160,240,360]
        elif dataset == 'cub':
            Ks = [160,240,360]
        else:
            Ks = [base+n_concepts_run*1, base+n_concepts_run*2, base+n_concepts_run*3, base+n_concepts_run*4, base+n_concepts_run*5,base+n_concepts_run*10]
        runs = get_activations(argo_seeds, dataset, acq_fns, kernels, Ks, results_folder)
        combinations += list(runs)


    # Get combination for this SLURM array task
    idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
    dataset = combinations[idx]

    time_date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    outfile = f"{os.environ['FAST']}/results/logs/ARGO-CBM_{time_date}_{dataset['dataset']}_{dataset['seed']}_{dataset['K']}.txt"

    # Set env vars for the experiment
    env = os.environ.copy()


    # Run the experiment
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as f:
        #import pdb; pdb.set_trace()
        if dataset['dataset'] == 'cub':
            result = subprocess.run(["python", 
                                    "train.py", 
                                    "-m", "argo",
                                    "-unfreeze", "0",
                                    "-val_interval", "5",
                                    "-patience", "3",
                                    "-n_epochs", "100",
                                    "-num_workers", "8",
                                    "-batch_size", "2048",
                                    "-pool_size", str(dataset['K']),
                                    "-d", dataset['dataset'],
                                    "-backbone", "resnet18_cub",
                                    "-seed", str(dataset['seed']),
                                    "-argo_train", dataset['train'],
                                    "-argo_val", dataset['val'],
                                    "-loss_fn", "ce",
                                    ], env=env, capture_output=True, text=True)
        else:
            result = subprocess.run(["python", 
                                    "train.py", 
                                    "-m", "argo",
                                    "-unfreeze", "100",
                                    "-val_interval", "5",
                                    "-patience", "3",
                                    "-n_epochs", "200",
                                    "-num_workers", "8",
                                    "-batch_size", "2048",
                                    "-pool_size", str(dataset['K']),
                                    "-d", dataset['dataset'],
                                    "-seed", str(dataset['seed']),
                                    "-argo_train", dataset['train'],
                                    "-argo_val", dataset['val'],
                                    "-loss_fn", "ce",
                                    ], env=env, capture_output=True, text=True)
        
        f.write(result.stdout)
        if result.stderr:
            f.write("\nERRORS:\n")
            f.write(result.stderr)

except Exception as e:
    time_date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    logfile = f"{os.environ['FAST']}/results/logs/logs_{time_date}_ID{idx}.txt"
    #traceback.print_exc()
    tb = traceback.format_exc()
with open(logfile, "w") as f:
    f.write(tb)

END
