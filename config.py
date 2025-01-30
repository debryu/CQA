SAVED_MODELS_FOLDER = {  
  "vlgcbm":"./saved_models/",
  "lfcbm":"./saved_models/",
  "labo":"./saved_models/",
  "resnetcbm":"./saved_models/",
  "llamaoracle":"./saved_models/",
  #"lfcbm":"./models/LFC/saved_models/",
}

ACTIVATIONS_PATH = {
    "shared":"./data/activations/",   # Share the activations between models and runs to save space
    "default":"",                     # Save the activation in each of the model folders
}

DATASETS_FOLDER_PATHS = {
  "celeba":"/mnt/cimec-storage6/users/nicola.debole/home/data/celeba_manual_download",
  "shapes3d":"./data/shapes3d/",
  "cifar10":"./data/cifar10/",
  "cub":"./data/cub/"
}

CONCEPT_SETS = {
  "celeba":"./data/concepts/celeba/handmade.txt",
  "shapes3d":"./data/concepts/shapes3d/shapes3d.txt",
  "cifar10": "./data/concepts/cifar10/cifar10_filtered.txt",
}

LLM_GENERATED_ANNOTATIONS = "./data/llava-phi3_annotations"

LABELS = {
  "celeba": ['male', 'female'],
  "cifar10": ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
  "shapes3d": ['red pill', 'not a red pill'],
}

def folder_naming_convention(args):
  ''' Naming convention for the saved model
  Available flags:
  '''
  return f"{args.model}_{args.dataset}_{args.date}_{args.time}"
