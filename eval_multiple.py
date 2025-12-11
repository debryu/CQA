import subprocess
import os
import itertools
from loguru import logger
from concurrent.futures import ProcessPoolExecutor
import datetime
from multiprocessing import Pool, Manager, Lock
import time



model_to_eval = [
    # Competitors
    # celeba
    '/mnt/cimec-storage6/users/nicola.debole/home/CQA/ordered_models/celeba/resnetcbm/resnetcbm_celeba_2025_02_24_17_44',
    '/mnt/cimec-storage6/users/nicola.debole/home/CQA/ordered_models/celeba/labo/labo_celeba_2025_03_10_10_31',
    '/mnt/cimec-storage6/users/nicola.debole/home/CQA/ordered_models/celeba/lfcbm/lfcbm_celeba_2025_02_04_13_37',
    '/mnt/cimec-storage6/users/nicola.debole/home/CQA/ordered_models/celeba/vlgcbm/vlgcbm_celeba_2025_02_25_17_23',
    
    # cub
    '/mnt/cimec-storage6/users/nicola.debole/home/CQA/ordered_models/cub/resnetcbm/resnetcbm_cub_2025_03_08_19_04',
    '/mnt/cimec-storage6/users/nicola.debole/home/CQA/ordered_models/cub/labo/labo_cub_2025_03_08_20_29',
    '/mnt/cimec-storage6/users/nicola.debole/home/CQA/ordered_models/cub/lfcbm/lfcbm_cub_2025_03_08_20_41',
    '/mnt/cimec-storage6/users/nicola.debole/home/CQA/ordered_models/cub/vlgcbm/vlgcbm_cub_2025_03_10_00_57',
    
    # shapes3d
    '/mnt/cimec-storage6/users/nicola.debole/home/CQA/ordered_models/shapes3d/resnetcbm/resnetcbm_shapes3d_2025_03_10_00_57',
    '/mnt/cimec-storage6/users/nicola.debole/home/CQA/ordered_models/shapes3d/labo/labo_shapes3d_2025_03_10_10_44',
    '/mnt/cimec-storage6/users/nicola.debole/home/CQA/ordered_models/shapes3d/lfcbm/lfcbm_shapes3d_2025_02_21_16_39',
    '/mnt/cimec-storage6/users/nicola.debole/home/CQA/ordered_models/shapes3d/vlgcbm/vlgcbm_shapes3d_2025_02_25_20_30',
    
    # derma
    '/mnt/cimec-storage6/users/nicola.debole/home/CQA/saved_models/lfcbm_dermamnist_2025_10_29_10_15',
    '/mnt/cimec-storage6/users/nicola.debole/home/CQA/saved_models/labo_dermamnist_2025_10_29_10_22',
    '/mnt/cimec-storage6/users/nicola.debole/home/CQA/saved_models/resnetcbm_dermamnist_2025_10_27_10_53',
    
    
    # Argo celeba
    '/mnt/cimec-storage6/users/nicola.debole/home/CQA/saved_models/argus_celeba_2025_10_17_22_03',
    '/mnt/cimec-storage6/users/nicola.debole/home/CQA/saved_models/argus_celeba_2025_10_17_22_02',
    '/mnt/cimec-storage6/users/nicola.debole/home/CQA/saved_models/argus_celeba_2025_10_17_12_34',
    '/mnt/cimec-storage6/users/nicola.debole/home/CQA/saved_models/argus_celeba_2025_10_17_12_36',
    
    # Argo cub
    '/mnt/cimec-storage6/users/nicola.debole/home/CQA/saved_models/argus_cub_2025_11_03_10_02',
    '/mnt/cimec-storage6/users/nicola.debole/home/CQA/saved_models/argus_cub_2025_09_29_09_48',
    '/mnt/cimec-storage6/users/nicola.debole/home/CQA/saved_models/argus_cub_2025_09_03_17_16',
    '/mnt/cimec-storage6/users/nicola.debole/home/CQA/saved_models/argus_cub_2025_09_01_14_32',
    
    # Argo derma
    '/mnt/cimec-storage6/users/nicola.debole/home/CQA/saved_models/argus_dermamnist_2025_10_23_14_45',
    '/mnt/cimec-storage6/users/nicola.debole/home/CQA/saved_models/argus_dermamnist_2025_10_23_14_46',
    '/mnt/cimec-storage6/users/nicola.debole/home/CQA/saved_models/argus_dermamnist_2025_10_23_17_13',
    '/mnt/cimec-storage6/users/nicola.debole/home/CQA/saved_models/argus_dermamnist_2025_10_23_17_49',
    
    # Argo shapes3d
    '/mnt/cimec-storage6/users/nicola.debole/home/CQA/saved_models/argus_shapes3d_2025_10_09_09_51',
    '/mnt/cimec-storage6/users/nicola.debole/home/CQA/saved_models/argus_shapes3d_2025_10_08_16_02',
    '/mnt/cimec-storage6/users/nicola.debole/home/CQA/saved_models/argus_shapes3d_2025_10_08_11_24',
    '/mnt/cimec-storage6/users/nicola.debole/home/CQA/saved_models/argus_shapes3d_2025_09_29_09_41',
    
    # CBM %227 dermamnist
    '/mnt/cimec-storage6/users/nicola.debole/home/CQA/saved_models/cbmlite_dermamnist_2025_10_27_11_02',
]

import subprocess


for f in model_to_eval:
    print(f"Running with {f}...")
    subprocess.run(["python", "main.py", "-load_dir", f, "-all"])
