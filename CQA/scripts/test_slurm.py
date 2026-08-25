import os, itertools, subprocess, datetime
from CQA.utils.load_ARGO_activations import get_activations
import traceback

# Find all experiments

path = "results/ICML/CBM/competitors"

FOLDER = os.path.join(os.environ['FAST'], path)

files = os.listdir(FOLDER)
experiments = []
for f in files:
    if f.endswith(".txt"):
        continue
    if "SEED" not in f:
        if "competitors" in path:
            experiments.append(f)
        continue
    else:
        experiments.append(f)

print(experiments)
ads
idx = 3
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
                            "-concept_metrics",
                            "-force",
                            ], capture_output=True, text=True)
    f.write(result.stdout)
    if result.stderr:
        f.write("\nERRORS:\n")
        f.write(result.stderr)

