SAVED_MODELS_FOLDER = {  
  "vlgcbm":"/leonardo_scratch/fast/IscrC_ARGO/results/CBMs",
  "lfcbm":"/leonardo_scratch/fast/IscrC_ARGO/results/CBMs",
  "labo":"/leonardo_scratch/fast/IscrC_ARGO/results/CBMs",
  "resnetcbm":"/leonardo_scratch/fast/IscrC_ARGO/results/CBMs",
  "oracle":"/leonardo_scratch/fast/IscrC_ARGO/results/CBMs",
  "argo":"/leonardo_scratch/fast/IscrC_ARGO/results/CBMs"
  #"lfcbm":"./models/LFC/saved_models/",
}

ACTIVATIONS_PATH = {
  "shared":"./data/activations/",
    #"shared":"/mnt/cimec-storage6/shared/assembly/data/activations/",   # Share the activations between models and runs to save space
    "default":"",                     # Save the activation in each of the model folders
}

DATASETS_FOLDER_PATHS = {
  "celeba":"/leonardo/home/userexternal/ndebole0/data/celeba_manual_download" ,
  "shapes3d":"/leonardo/home/userexternal/ndebole0/data/shapes3d" ,
  #"cifar10":"/mnt/cimec-storage6/shared/cv_datasets/cifar10",
  "cub":"/leonardo/home/userexternal/ndebole0/data/cub",
  "dermamnist":'/leonardo/home/userexternal/ndebole0/data/dermamnist',
}

CONCEPT_SETS = {
  "root":"./data_concepts/",
  "celeba":"/leonardo_scratch/fast/IscrC_ARGO/concepts/celeba/concepts.txt",
  "shapes3d":"/leonardo_scratch/fast/IscrC_ARGO/concepts/shapes3d/concepts.txt",
  #"cifar10": "/leonardo_scratch/fast/IscrC_ARGO/concepts/celeba/concepts.txt",
  "cub":"/leonardo_scratch/fast/IscrC_ARGO/concepts/cub/concepts.txt",
  "cub_short":"/leonardo_scratch/fast/IscrC_ARGO/concepts/cub/short.txt",
  "dermamnist":"/leonardo_scratch/fast/IscrC_ARGO/concepts/dermamnist/concepts.txt",
}

LLM_GENERATED_ANNOTATIONS = "./data/llava-phi3_annotations"
DINO_GENERATED_ANNOTATIONS = "./data/VLG_annotations/new_anno"

CLASSES = {
  'cub':'./data/concepts/cub/classes.txt',
  'celeba': './data/concepts/celeba/classes.txt',
  'shapes3d': './data/concepts/shapes3d/classes.txt',
}

LABELS = {
  "celeba": ['male', 'female'],
  "cifar10": ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
  "shapes3d": ['red pill', 'not a red pill'],
  "cub": list(range(200)),
  "dermamnist": list(range(2)),
}

def folder_naming_convention(args):
  ''' Naming convention for the saved model
  Available flags:
  '''
  return f"{args.model}_{args.dataset}_{args.date}_{args.time}_SEED={args.seed}"

METRICS = ['label_accuracy', 'label_f1','disentanglement', 'concept_accuracy', 'avg_concept_accuracy', 'avg_concept_f1', 'ois', 'leakage',
           'avg_concept_auc','concept_auc']

'''#####################################
   ###       AVAILABLE METRICS       ###
   #####################################

    LABEL RELATED:
    - label_accuracy
    - label_f1

    CONCEPT RELATED:
    - avg_concept_accuracy
    - avg_concept_f1
    - concept_accuracy
    - concept_f1
    - concept_classification_reports
    
    DCI RELATED:
    - disentanglement
    - completeness
'''

REQUIRES_SIGMOID = ['labo', 'lfcbm']

SPLIT_INDEXES = {
  'cub_train':[0,4796],
  'cub_val':[0,1198],
  'shapes3d_train':[0,48000],
  'shapes3d_val':[0,5000],
  'celeba_train':[25000,50000],
  'celeba_val':[0,5000],
}

# Put here the datasets you have implemented that do not have concepts
CONCEPTS_NOT_AVAILABLE = ["cifar10", "cifar100"]
