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
  "celeba":"C:\\Users\\debryu\\Desktop\\VS_CODE\\HOME\\ML\\work\\VLG-CBM\\datasets\\celeba_manual_download",
  "shapes3d":"./data/shapes3d/",
  "cifar10":"./data/cifar10/",
  "cub":"./data/cub/"
}

CONCEPT_SETS = {
  "celeba":"./data/concepts/celeba/handmade.txt",
  "shapes3d":"./data/concepts/shapes3d/shapes3d.txt",
  "cifar10": "./data/concepts/cifar10/cifar10_filtered.txt",
  "cub":"./data/concepts/cub/cub_preprocess.txt",
}

LLM_GENERATED_ANNOTATIONS = "./data/llava-phi3_annotations"

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

METRICS = ['label_accuracy', 'label_f1','disentanglement', 'concept_accuracy']

'''#####################################
   ###       AVAILABLE METRICS       ###
   #####################################

    LABEL RELATED:
    - label_accuracy
    - label_f1

    CONCEPT RELATED:
    - avg_concept_accuracy
    - concept_accuracy
    - concept_f1
    - concept_classification_reports
    
    DCI RELATED:
    - disentanglement
    - completeness
'''