SAVED_MODELS_FOLDER = {  
  "vlgcbm":"./saved_models/",
  "lfcbm":"./saved_models/",
  "labo":"./saved_models/",
  "resnetcbm":"./saved_models/",
  "oracle":"./saved_models/",
  #"lfcbm":"./models/LFC/saved_models/",
}

ACTIVATIONS_PATH = {
    "shared":"/mnt/cimec-storage6/shared/assembly/data/activations/",   # Share the activations between models and runs to save space
    "default":"",                     # Save the activation in each of the model folders
}

DATASETS_FOLDER_PATHS = {
  "celeba":"/mnt/cimec-storage6/shared/cv_datasets/celeba_manual_download" ,
  "shapes3d":"/mnt/cimec-storage6/shared/cv_datasets/shapes3d" ,
  #"cifar10":"/mnt/cimec-storage6/shared/cv_datasets/cifar10",
  "cub":"/mnt/cimec-storage6/shared/cv_datasets/cub/"
}

CONCEPT_SETS = {
  "root":"/mnt/cimec-storage6/shared/assembly/data/concepts/",
  "celeba":"/mnt/cimec-storage6/shared/assembly/data/concepts/celeba/handmade.txt",
  "shapes3d":"/mnt/cimec-storage6/shared/assembly/data/concepts/shapes3d/shapes3d.txt",
  "cifar10": "/mnt/cimec-storage6/shared/assembly/data/concepts/cifar10/cifar10_filtered.txt",
  "cub":"/mnt/cimec-storage6/shared/assembly/data/concepts/cub/cub_preprocess.txt",
}

LLM_GENERATED_ANNOTATIONS = "/mnt/cimec-storage6/shared/assembly/data/llava-phi3_annotations"

CLASSES = {
  'cub':'/mnt/cimec-storage6/shared/assembly/data/concepts/cub/classes.txt',
  'celeba': '/mnt/cimec-storage6/shared/assembly/data/concepts/celeba/classes.txt',
  'shapes3d': '/mnt/cimec-storage6/shared/assembly/data/concepts/shapes3d/classes.txt',
}

LABELS = {
  "celeba": ['male', 'female'],
  "cifar10": ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
  "shapes3d": ['red pill', 'not a red pill'],
  "cub": list(range(200)),
}

def folder_naming_convention(args):
  ''' Naming convention for the saved model
  Available flags:
  '''
  return f"{args.model}_{args.dataset}_{args.date}_{args.time}"

METRICS = ['label_accuracy', 'label_f1','disentanglement', 'concept_accuracy', 'avg_concept_accuracy', 'avg_concept_f1']

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
}