import os, itertools, subprocess, datetime
#from CQA.utils.load_ARGO_activations import get_activations
import traceback
try:
    # Find all experiments
    FOLDER = os.path.join(os.environ['FAST'],"results/CBMs")

    files = os.listdir(FOLDER)
    experiments = []
    for f in files:
        if f.endswith(".txt"):
            continue
        if "SEED" not in f:
            continue
        experiments.append(f)

    print(len(experiments), "EXPERIMENTS")
    asd
    for e in experiments:
        load_dir = f"{os.environ['FAST']}/results/CBMs/{e}"
        outfile = f"{os.environ['FAST']}/results/CBMs/{e}.txt"
        # Run the experiment
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        with open(outfile, "w") as f:
            result = subprocess.run(["python", 
                                    "main.py", 
                                    "-load_dir", load_dir,
                                    "-label_metrics",
                                    "-dci", 
                                    "-concept_metrics"
                                    ], capture_output=True, text=True)
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