import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.temp_hinton import hinton
import pickle
from loguru import logger
from config import CONCEPT_SETS

VOCAB = {
    'betaglancenet': 'BGN',
    'cbmbase': 'CBM',
    'shapes3d': 'S3D',
    'celeba': 'CEL',

}

SHAPES3D_GROUND_TRUTH = [
  "FLOOR: red",
  "FLOOR: orange",
  "FLOOR: yellow",  
  "FLOOR: green",
  "FLOOR: lime",
  "FLOOR: azure",
  "FLOOR: blue",
  "FLOOR: navy",
  "FLOOR: purple",
  "FLOOR: pink",
  "WALL: red",
  "WALL: orange",
  "WALL: yellow",  
  "WALL: green",
  "WALL: lime",
  "WALL: azure",
  "WALL: blue",
  "WALL: navy",
  "WALL: purple",
  "WALL: pink",
  "OBJ: red",
  "OBJ: orange",
  "OBJ: yellow",  
  "OBJ: green",
  "OBJ: lime",
  "OBJ: azure",
  "OBJ: blue",
  "OBJ: navy",
  "OBJ: purple",
  "OBJ: pink",
  "SIZE: 0",
  "SIZE: 1",
  "SIZE: 2",
  "SIZE: 3",
  "SIZE: 4",
  "SIZE: 5",
  "SIZE: 6",
  "SIZE: 7",
  "SHAPE: cube",
  "SHAPE: cylinder",
  "SHAPE: sphere",
  "SHAPE: pill",
]
CELEBA_GROUND_TRUTH = [
  "5_o_Clock_Shadow",
  "Arched_Eyebrows",
  "Attractive",
  "Bags_Under_Eyes",
  "Bald",
  "Bangs",
  "Big_Lips",
  "Big_Nose",
  "Black_Hair",
  "Blond_Hair",
  "Blurry",
  "Brown_Hair",
  "Bushy_Eyebrows",
  "Chubby",
  "Double_Chin",
  "Eyeglasses",
  "Goatee",
  "Gray_Hair",
  "Heavy_Makeup",
  "High_Cheekbone",
  "Mouth_Sl_Open",
  "Mustache",
  "Narrow_Eyes",
  "No_Beard",
  "Oval_Face",
  "Pale_Skin",
  "Pointy_Nose",
  "Reced_Hairline",
  "Rosy_Cheeks",
  "Sideburns",
  "Smiling",
  "Straight_Hair",
  "Wavy_Hair",
  "Wearing_Earrings",
  "Wearing_Hat",
  "Wearing_Lipstick",
  "Wearing_Necklace",
  "Wearing_Necktie",
  "Young"
]
CHESTMNIST_GROUND_TRUTH = [
  "Atelectasis",
  "Cardiomegaly",
  "Effusion",
  "Infiltration",
  "Mass",
  "Nodule",
  "Pneumonia",
  "Pneumothorax",
  "Consolidation",
  "Edema",
  "Emphysema",
  "Fibrosis",
  "Pleural_Thickening",
  "Hernia"
]

with open(CONCEPT_SETS['cub']) as f:
        cub_concepts = f.read().split("\n")

GROUND_TRUTH_CONCEPTS = {
    "celeba": CELEBA_GROUND_TRUTH,
    "shapes3d": SHAPES3D_GROUND_TRUTH,
    "chestmnist": CHESTMNIST_GROUND_TRUTH,
    "cub": cub_concepts,
}

def save_IM_as_img(save_path,name,title,importance_matrix,save_plot=True):
  dim1,dim2 = importance_matrix.shape
  heatmap(importance_matrix,(dim1,dim2),title, save_plot=save_plot, save_path=os.path.join(save_path,name))
  #visualise(save_path, name, importance_matrix, (dim1,dim2), title, save_plot=save_plot)
  
  return 

def heatmap(matrix, dataset_name, plot_title, save_path=None):
    try:
      GROUND_TRUTH_CONCEPTS[dataset_name]
    except:
      logger.error("Please add ground truth concepts for the dataset in utils/dci_utils.py")
      raise ValueError(f"Dataset {dataset_name} not found in GROUND_TRUTH_CONCEPTS")
    with open(CONCEPT_SETS[dataset_name]) as f:
        concepts = f.read().split("\n")
    y_labels = [c for c in concepts]
    plt.figure(figsize=matrix.shape)
    ax = sns.heatmap(matrix, annot=False, 
                fmt=".2f", 
                cmap="coolwarm", 
                linewidths=0.5, 
                square=True, 
                xticklabels=GROUND_TRUTH_CONCEPTS[dataset_name],
                yticklabels=y_labels,
                cbar=True)
  
    plt.xticks(rotation=50, ha='right', va='top', )
    plt.title(plot_title, fontsize=80, pad=20)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

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


