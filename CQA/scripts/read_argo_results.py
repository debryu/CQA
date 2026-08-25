import os
FAST_STORAGE = os.environ["FAST"]
folders = os.listdir(os.path.join(FAST_STORAGE, "results/ARGOCBM")) 
for folder in folders:
    if os.path.exists(os.path.join(FAST_STORAGE,folder,"metrics.txt"))
    print(folder)
    
