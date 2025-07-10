import os
import pickle
from loguru import logger
from matplotlib import pyplot as plt
dataset = 'cub'
folder = f'./ordered_models/{dataset}'
import numpy as np
subfolders = os.listdir(folder)
shapes = ['p','s','o','^','v']

import matplotlib
from matplotlib import rc
#rc('font', **{'family':'serif', 'serif':['Computer Modern Roman'], 
#                                'monospace': ['Computer Modern Typewriter']})
params = {'backend': 'pdf',
          'axes.labelsize': 12,
          #'text.fontsize': 12,
          'legend.fontsize': 12,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'text.usetex': False,
          'figure.figsize': (6, 4),
          'axes.unicode_minus': True}
#matplotlib.rcParams.update(params)


model_translation = {
    'labo':r"LaBo",
    'lfcbm':r'LF-CBM',
    'oracle':r'$Naive$',
    'resnetcbm':r'CBM',
    'vlgcbm':r'VLG-CBM',
    
}

model_linestyle = {
    'labo':"-",
    'lfcbm':"-",
    'oracle':"-",
    'resnetcbm':"-",
    'vlgcbm':":",
    
}
def plot_delta(deltas):
    folder = './'
    # Plot
    plt.figure(figsize=(6, 4))
    if dataset == 'celeba':
        colors = ['#264653','#f4a261','#2a9d8f','#e9c46a','#e76f51']
    else:
        colors = ['#264653','#f4a261','#2a9d8f','#e9c46a','#e76f51']
    for i,model in enumerate(deltas.keys()):
        array = np.array(deltas[model])
        # Compute mean along axis=0 (column-wise mean)
        mean_values = np.mean(array, axis=0)
        std_values = np.std(array,axis=0)
        #d = mean_values
        
        x = range(len(mean_values))
        #print(len(accuracies_pred))
        #plt.plot(x, accuracies_gt, marker='o', linestyle='-', label="Acc gt", color = '#00ccff')
        #plt.plot(x, d, marker='o', linestyle='-', label=model, color = colors[i])
        # Plot with error bars
        #plt.errorbar(x, mean_values, yerr=std_values, fmt='o-', label=model_translation[model], color=colors[i], markersize=2, capsize=3, capthick=1, alpha=0.9)
        plt.plot(x, mean_values, marker=shapes[i], linestyle=model_linestyle[model], label=model_translation[model], color=colors[i], markersize=3, alpha=0.9)
        plt.fill_between(x, mean_values - std_values, mean_values + std_values, color=colors[i], alpha=0.3)

        #plt.plot(x, losses, marker='o', linestyle='-', label="Values", color = 'red')
        #plt.plot(x, I_gt, marker='o', linestyle='-', label="1-H(y|gt)/H(y)", color = '#1D6D47')
        #plt.plot(x, I_pred, marker='o', linestyle='-', label="1-H(y|c)/H(y)", color = '#993333')
        #print(len(concepts))
        #print(len(indices))
        # Set custom labels
        #print(indices)
        #print(indices_inc)
        #ne_wss = indices[indices_inc]
        #print(ne_wss)
        
    plt.xlabel("Number of Concepts")
    plt.ylabel("Gap")
    #plt.title("Plot of Tensor Values")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)  # Add horizontal line at y=0
    from matplotlib import font_manager
    plt.legend()
    #plt.legend(fontsize=12, prop=font_manager.FontProperties(family='Computer Modern Typewriter'))
    plt.grid(True)
    # Show plot
    plt.savefig(os.path.join(folder, "leakplot.png"), dpi=300, bbox_inches="tight")


leakage_deltas = {'resnetcbm':[],'oracle':[],'labo':[],'lfcbm':[],'vlgcbm':[]}


# Optional set up all the folders to check 
subfolders = ["./ordered_models/shapes3d/labo",
              "./ordered_models/shapes3d/lfcbm",
              "./ordered_models/shapes3d/resnetcbm",
              "./ordered_models/shapes3d/vlgcbm",
              "./ordered_models/tentative/shapes3d",
              ]
folder = ""

for f in subfolders:
    models = os.listdir(os.path.join(folder,f))
    for m in models:
        model = m.split("_")[0]
        
        try:
            # Load leakage
            with open(os.path.join(folder,f,m,"leakage.data"), "rb") as file:
                leakage = pickle.load(file)            
            delta = leakage['delta']
            #print(delta)
            leakage_deltas[model].append(delta)
        except:
            logger.warning(f"Missing {os.path.join(folder,f,m)}")
        
        
        
#print(leakage_deltas)
plot_delta(leakage_deltas)

