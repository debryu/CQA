


import subprocess
import itertools
import os
from multiprocessing import Pool
from loguru import logger

CUDA_VISIBLE_DEVICES = '2'
vlg_run = {
    # GRID SEARCH PARAMETERS
    "-cbl_epochs": [30,
                    ],
    "-cbl_confidence_threshold":[0.15],
    "-cbl_lr": [0.01],
    "-crop_to_concept_prob": [0.0, 0.3, 0.5],

    # FIXED PARAMETERS
    "-model": ["vlgcbm"],
    "-dataset": ["shapes3d",
                ],
    "-cbl_optimizer": ["adam"],
    "-skip_concept_filter":[""],
    "-wandb":[""],
    
}

resnetcbm_run = {
    # GRID SEARCH PARAMETERS
    "-epochs": [20,
                    ],
    "-unfreeze":[5],
    "-lr": [0.001],
    "-balanced":[""],
    "-dropout_prob":[0.01],

    # FIXED PARAMETERS
    "-val_interval":[1],
    "-model": ["resnetcbm"],
    "-dataset": [
                    "celeba",
                    "shapes3d"
                ],
    "-wandb":[""],
    "-seed":["64","65","66","67","68"],
}

lfcbm_run = {
    # GRID SEARCH PARAMETERS
    

    # FIXED PARAMETERS
    "-model": ["lfcbm"],
    "-dataset": [
                    #"celeba",
                    "shapes3d"
                ],
    "-wandb":[""],
    "-seed":["42"],
}

labo_run = {
    # GRID SEARCH PARAMETERS
    

    # FIXED PARAMETERS
    "-model": ["labo"],
    "-dataset": [
                    #"celeba",
                    "shapes3d"
                ],
    "-wandb":[""],
    "-seed":["42"],
}

runs = [lfcbm_run]


import os

for run in runs:
    # Iterate over all combinations of parameters
    for combination in itertools.product(*run.values()):
        command = f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} python train.py"
        for i,key in enumerate(run.keys()):
                    command += f" {key}"
                    if str(combination[i]) != '':
                        command += f" {combination[i]}"
        logger.info(f"Running: {command}")  # Print the command for debugging
        os.system(command)  # Runs as if manually executed in terminal


'''
env = os.environ.copy()  # Copy existing environment variables
env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES  # Add a new variable

def run_grid_search(param_set):
    command = ["python","-u","train.py",
               "-m", "vlgcbm",
               "-d", "celeba",
               "-cbl_optimizer", "adam",
               "-skip_concept_filter",
               "-wandb",
               f"-cbl_epochs={param_set[0]}",
               f"-cbl_confidence_threshold={param_set[1]}",
               f"-crop_to_concept_prob={param_set[2]}",
               f"-cbl_lr={param_set[3]}",
               
               ]
    
    
    print(f"Running: {command}")  # Print the command for debugging
    res = subprocess.run(command, env=env, text=True, capture_output=True)  # Run the command
    return res

def run_grid_search_resnet(param_set):
    command = ["python","-u","train.py",
               "-m=resnetcbm",
               "-d=celeba",
               "-wandb",
               f"-e={param_set[0]}",
               f"-dropout_prob={param_set[2]}",
               f"-lr={param_set[3]}",
               f"-unfreeze={param_set[1]}",
               
               ]
    
    
    print(f"Running: {command}")  # Print the command for debugging
    print(env['CUDA_VISIBLE_DEVICES'])
    res = subprocess.run(command, env=env, text=True, capture_output=True)  # Run the command
    
    for line in res.stdout:
            print(line, end="")
            res.wait()
            err_out = res.stderr.read()
            if err_out:
                print(err_out)
    return res

def create_grid(run):
    param_grid = []
    for key,value in run.items():
        print(value)
        if len(value) > 1:
            param_grid.append(key)
            param_grid.append(tuple(value))
        else:
            if value != '':
                param_grid.append(key)
    return param_grid

def asd():
    for run in runs:
        # Iterate over all combinations of parameters
        for combination in itertools.product(*run.values()):
            command = ["python","-u","train.py"]
            for i,key in enumerate(run.keys()):
                command.append(key)
                if str(combination[i]) != '':
                    command.append(str(combination[i]))
            print(f"Running: {command}")  # Print the command for debugging
            res = subprocess.run(command, env=env, text=True, capture_output=True)  # Run the command
            for line in res.stdout:
                print(line, end="")
            res.wait()
            err_out = res.stderr.read()
            if err_out:
                print(err_out)

with Pool(processes=1) as pool:
    #lr_to_epochs = {0.01: 3, 0.001:6, 0.0001:50}
    #threshold = [0.2,0.3,0.4,0.5,0.6,0.7]
    #crop_prob = [0.0,0.25,0.5]
    lr_to_epochs = {0.01: 3, 0.001:6, 0.0001:80}
    threshold = [1,3,5]
    crop_prob = [0.01,0.1,0.2]
    lr = [0.0001]
    combinations = itertools.product(threshold,crop_prob,lr)
    
    final_comb = [(lr_to_epochs[lr],thr,cp,lr) for thr, cp, lr in combinations]
    results = pool.map(run_grid_search_resnet, final_comb)
    for output in results:
        print(output)
        
'''


