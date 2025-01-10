from config import DATASETS_FOLDER_PATHS
import datasets.dataset_classes
import inspect
from loguru import logger
classes = {}
# Get all classes from the module
for name, cls in inspect.getmembers(dataset_classes, inspect.isclass):
    try:
      name = cls.name
    except:
      continue
    classes[name] = cls


logger.debug(f"Available datasets: {classes}")
#probe_dataset, probe_split, probe_dataset_root_dir, preprocess_fn, split_idxs=None
def get_dataset(ds_name,**kwargs):
  if ds_name not in DATASETS_FOLDER_PATHS:
    raise ValueError(f"Dataset {ds_name} not found in DATASETS_FOLDER_PATHS")
  return classes[ds_name](root = DATASETS_FOLDER_PATHS[ds_name], **kwargs)