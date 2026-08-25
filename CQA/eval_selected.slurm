#!/bin/bash
#SBATCH --job-name=eval_s
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=16:00:00
#SBATCH --array=0-5%2           # Adjust based on #combinations and concurrency
#SBATCH --output=/leonardo_scratch/fast/IscrC_ARGO/results/logs/eval-slurm-%A_%a.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_ARGO/results/logs/eval-slurm-%A_%a.err

module load python
module load cuda
source /leonardo/home/userexternal/ndebole0/projects/vlm/bin/activate

# Map SLURM_ARRAY_TASK_ID to your experiment combination
python - <<'END'
import os, itertools, subprocess, datetime
from CQA.utils.load_ARGO_activations import get_activations
import traceback
try:
    # Find all experiments
    path = "results/ARGOCBM"
    FOLDER = os.path.join(os.environ['FAST'],path)

    
    experiments = [
        "celeba-40-randomc-RBF-123_2026_02_21_19_34_SEED=123",
        "celeba-100-randomc-RBF-123_2026_02_21_19_34_SEED=123",
        "celeba-160-randomc-RBF-123_2026_02_21_19_34_SEED=123",
        "celeba-220-randomc-RBF-123_2026_02_21_19_34_SEED=123",
        "celeba-280-randomc-RBF-123_2026_02_21_19_34_SEED=123",
    ]
    

    idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
    e = experiments[idx]
    load_dir = f"{os.environ['FAST']}/{path}/{e}"
    outfile = f"{os.environ['FAST']}/{path}/{e}.txt"
    # Run the experiment
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as f:
        result = subprocess.run(["python", 
                                "main.py", 
                                "-load_dir", load_dir,
                                "-label_metrics",
                                "-dci", 
                                "-force",
                                "-concept_metrics"
                                ], capture_output=True, text=True)
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
