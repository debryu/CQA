import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.temp_hinton import hinton
import pickle

VOCAB = {
    'betaglancenet': 'BGN',
    'cbmbase': 'CBM',
    'shapes3d': 'S3D',
    'celeba': 'CEL',

}

def save_IM_as_img(save_path,name,title,importance_matrix,save_plot=True):
  dim1,dim2 = importance_matrix.shape
  visualise(save_path, name, importance_matrix, (dim1,dim2), title, save_plot=save_plot)
  #plt.savefig(save_path + "/importance_matrix.png")
  return 

def visualise(save_path, name, R, dims, title = 'plot', save_plot=False):
    # init plot (Hinton diag)
    x,y = dims
    fig, axs = plt.subplots(1, figsize=(x, y), facecolor='w', edgecolor='k')
    
    # visualise
    hinton(R, 'Z', 'C', ax=axs, fontsize=10)
    axs.set_title('{0}'.format(title), fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.rc('text', usetex=False)
    if save_plot:
        fig.tight_layout()
        plt.savefig(os.path.join(save_path,name))