from metrics.dci import DCI_wrapper
from utils.dci_utils import save_IM_as_img
import pickle
import os
import json
#from utils.utils import compute_concept_frequencies
from models import get_model
from loguru import logger
from sklearn.metrics import classification_report as cr
from metrics.common import get_conceptWise_metrics
from config import LABELS
from utils.args_utils import load_args

# TODO: Fix TEMP and add the correct target names
# TODO: Count Imbalances for each ds once
class CONCEPT_QUALITY():
  def __init__(self, model):
    self.model = model
    self.output = None
    self.dci = None
    self.classification_report = None
    self.CQA_save_path = os.path.join(self.model.args.load_dir ,'CQA.pkl')
    if not os.path.exists(os.path.join(self.model.args.load_dir,"train_concept_freq.txt")) or self.model.args.force:
        #self.concept_freq,self.label_freq = compute_concept_frequencies(self.model.args.dataset)
        #json.dump(self.concept_freq, open(os.path.join(self.model.args.load_dir,"train_concept_freq.txt"), 'w'))
        #json.dump(self.label_freq, open(os.path.join(self.model.args.load_dir,"train_label_freq.txt"), 'w'))
        pass
    else:
        self.concept_freq = json.load(open(os.path.join(self.model.args.load_dir,"train_concept_freq.txt")))
        self.label_freq = json.load(open(os.path.join(self.model.args.load_dir,"train_label_freq.txt")))
    
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

  def get_classification_report(self):
    y_true = self.output['labels_gt'] 
    y_pred = self.output['labels_pred'].argmax(axis=1)
    target_names = LABELS[self.model.args.dataset]
    self.classification_report = cr(y_true, y_pred, target_names=target_names, output_dict=True)
    self.save()
    return self.classification_report

  def metrics(self):
    get_conceptWise_metrics(self.output, theshold=0.5)
    return
  
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

  def save_im_as_img(self, path,file_name, plot_title):
    save_IM_as_img(path, file_name, plot_title, self.dci['importance_matrix'])
    return 

def initialize_CQA(folder_path, args, split = 'test', force_from_scratch = False):
  logger.debug(f"Initializing CQA from {folder_path}")
  # Check if CQA (Concept Quality Analysis) is already present
  if os.path.exists(folder_path + '/CQA.pkl') and not force_from_scratch:
    logger.info("CQA found. Loading CQA.")
    with open(folder_path + '/CQA.pkl', 'rb') as f:
      CQA = pickle.load(f)
  else:
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