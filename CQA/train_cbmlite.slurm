#!/bin/bash
#SBATCH --job-name=train_cbmlite
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=16:00:00
#SBATCH --array=0-5%5            # Adjust based on #combinations and concurrency
#SBATCH --output=/leonardo_scratch/fast/IscrC_ARGO/results/logs/train-slurm-%A_%a.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_ARGO/results/logs/train-slurm-%A_%a.err

module load python
module load cuda
source /leonardo/home/userexternal/ndebole0/projects/vlm/bin/activate


# Map SLURM_ARRAY_TASK_ID to your experiment combination
python - <<'END'
import os, itertools, subprocess, datetime
import traceback
try:
    datasets = ['celeba']
    seeds = [5]#,8,13,21]
    n_concepts_run = 7
    base = 42
    if datasets[0] == 'dermamnist':
        Ks = [49,63,77,112,224]
    elif datasets[0] == 'celeba':
        Ks = [78,117,156,195,390]
    elif datasets[0] == 'shapes3d':
        Ks = [84,126,168,210,420]
    elif datasets[0] == 'cub':
        Ks = [127,232,337,412,487]
    else:
        Ks = [base+n_concepts_run*1, base+n_concepts_run*2, base+n_concepts_run*3, base+n_concepts_run*4, base+n_concepts_run*5,base+n_concepts_run*10]
    combinations = list(itertools.product(seeds, datasets, Ks))


    # Get combination for this SLURM array task
    idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
    seed,ds_name,K = combinations[idx]

    time_date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    outfile = f"{os.environ['FAST']}/results/logs/CBMLITE_{time_date}_{ds_name}_{seed}_{K}.txt"

    # Set env vars for the experiment
    env = os.environ.copy()


    # Run the experiment
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as f:
        #import pdb; pdb.set_trace()
        if ds_name == 'cub':
            result = subprocess.run(["python", 
                                    "train.py", 
                                    "-m", "cbmlite",
                                    "-d", ds_name,
                                    "-backbone", "resnet18_cub",
                                    "-unfreeze", "5",
                                    "-val_interval", "4",
                                    "-patience", "4",
                                    "-seed", str(seed),
                                    "-subset_size", str(K)
                                    ], env=env, capture_output=True, text=True)
        else:
            result = subprocess.run(["python", 
                                    "train.py", 
                                    "-m", "cbmlite",
                                    "-unfreeze", "5",
                                    "-d", ds_name,
                                    "-val_interval", "4",
                                    "-patience", "4",
                                    "-seed", str(seed),
                                    "-subset_size", str(K)
                                    ], env=env, capture_output=True, text=True)
        
        f.write(result.stdout)
        if result.stderr:
            f.write("\nERRORS:\n")
            f.write(result.stderr)

except Exception as e:
    time_date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    logfile = f"{os.environ['FAST']}/results/logs/logs_{time_date}.txt"
    #traceback.print_exc()
    with open(logfile, "w") as f:
        f.write(str(e))

END
