from metrics.dci import DCI_wrapper
#from utils.save_importance_matrix import save_IM_as_img
import pickle
import os
from config import MODELS_FOLDER_PATHS
from utils.load_args import load_args
from models import get_model
from loguru import logger

class CONCEPT_QUALITY():
  def __init__(self, model):
    self.model = model
    self.output = None
    self.dci = None
    self.CQA_save_path = os.path.join(self.model.args.load_dir ,'CQA.pkl')

  def store_output(self, split = 'test'):
    self.output = self.model.run(split)
    logger.debug(f"Output stored in CQA object.")
    return

  def save(self):
    pickle.dump(self, open(self.CQA_save_path, "wb"))
    logger.info(f"Saved to {self.CQA_save_path}")
    return

  def eval():
    pass

  def DCI(self,train_test_ratio=0.7,max_samples:int = None, level = 'INFO'):
    # Split the data in train-test
    n = len(self.output['concepts_pred'])
    train_size = int(n*train_test_ratio)
    representation_train = self.output['concepts_pred'][:train_size]
    representation_val = self.output['concepts_pred'][train_size:]
    concept_gt_train = self.output['concepts_gt'][:train_size]
    concept_gt_val = self.output['concepts_gt'][train_size:]

    if max_samples != None:
      representation_train = representation_train[:max_samples]
      concept_gt_train = concept_gt_train[:max_samples]
      representation_val = representation_val[:max_samples]
      concept_gt_val = concept_gt_val[:max_samples]

    logger.debug(f"Computing DCI with train_test_ratio={train_test_ratio}...")
    dci = DCI_wrapper(representation_train, concept_gt_train, representation_val, concept_gt_val, level)
    dci['train_test_ratio'] = train_test_ratio
    self.dci = dci
    self.save()
    return dci
'''
  def save_im_as_img(self, path,file_name, plot_title):
    save_IM_as_img(path, file_name, plot_title, self.dci['importance_matrix'])
    return 
'''
def initialize_CQA(folder_path, args, split = 'test', force_from_scratch = False):
  logger.debug(f"Initializing CQA from {folder_path}")
  # Check if CQA (Concept Quality Analysis) is already present
  try:
    if force_from_scratch:
      raise FileNotFoundError
    with open(folder_path + '/CQA.pkl', 'rb') as f:
      CQA = pickle.load(f)
  except:
    logger.info("CQA not found. Initializing CQA from scratch.")
    # Load args
    logger.debug(f"Loading args from {folder_path}")
    args.load_dir = folder_path
    args = load_args(args)
    # Load model
    
    model = get_model(args)
    logger.debug(f"Model loaded: {model}")
    # args are uploaded in the model, so no need to pass them again
    CQA = CONCEPT_QUALITY(model)
    logger.info(f"Running the model on {model.args.dataset} {split}...")
    # Run the model to get all the outputs
    CQA.store_output(split)
    CQA.save()
    
    
  return CQA