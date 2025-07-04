import os
import importlib
from loguru import logger

def _get_all_trainers():
    return [model.split(".")[0] for model in os.listdir("models/training") if model.endswith(".py") and model != "__init__.py"]

trainers = {}
last_layer_trainers = {}
for model in _get_all_trainers():
  logger.debug(f"Loading training script for model {model}")
  mod = importlib.import_module(f"CQA.models.training.{model}")
  trainers[model] = getattr(mod, "train")
  last_layer_trainers[model] = getattr(mod, "train_last_layer")

logger.debug(f"Loaded trainers: {trainers}")

def get_trainer(args):
  logger.info(f"Getting trainer model {args.model}")
  return trainers[args.model]

def get_last_layer_trainer(args):
  logger.info(f"Getting trainer model {args.model}")
  return last_layer_trainers[args.model]