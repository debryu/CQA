SAVED_MODELS_FOLDER = {  
  "vlgcbm":"./saved_models/",
  "lfcbm":"./saved_models/",
  "resnetcbm":"./saved_models/",
  #"lfcbm":"./models/LFC/saved_models/",
}

ACTIVATIONS_PATH = {
    "shared":"./data/activations/",   # Share the activations between models and runs to save space
    "default":"",                     # Save the activation in each of the model folders
}

DATASETS_FOLDER_PATHS = {
  "celeba":"C:\\Users\\debryu\\Desktop\\VS_CODE\\HOME\\ML\\work\\VLG-CBM\\datasets\\celeba_manual_download",
  "shapes3d":"./data/shapes3d/",
}

CONCEPT_SETS = {
  "celeba":"./data/concepts/celeba/",
}

LABELS = {
  "celeba": ['male', 'female'],
}

def folder_naming_convention(args):
  ''' Naming convention for the saved model
  Available flags:
  '''
  return f"{args.model}_{args.dataset}_{args.date}_{args.time}"
