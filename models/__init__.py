import os
import importlib
from loguru import logger
##### Certificate Expired, try to override it
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def _get_all_models():
    return [model.split(".")[0] for model in os.listdir("models") if model.endswith(".py") and model != "__init__.py" and model != "base.py" and model != "template.py"]

#TODO: Add a function to get all trainers
def _get_all_trainers():
    return [model.split(".")[0] for model in os.listdir("models") if model.endswith(".py") and model != "__init__.py" and model != "base.py" and model != "base.py"]

models = {}
for model in _get_all_models():
  #logger.debug(f"Loading model {model}")
  model_class = importlib.import_module(f"models.{model}")
  mod = importlib.import_module(f"models.{model}")
  class_names = {x.lower():x for x in mod.__dir__()}[model]
  models[model] = getattr(mod, class_names)

logger.debug(f"Available models: {models}")

def get_model(args):
  logger.info(f"Getting model {args.model}")
  return models[args.model](args)
