from metrics.dci import DCI_wrapper
from utils.dci_utils import save_IM_as_img, heatmap
import pickle
import torch
import os
import json
import copy
import numpy as np
#from utils.utils import compute_concept_frequencies
from models import get_model
from loguru import logger
from sklearn.metrics import classification_report as cr
from metrics.common import get_conceptWise_metrics, compute_AUCROC_concepts
from utils.eval_models import train_LR_on_concepts
from config import LABELS, METRICS, REQUIRES_SIGMOID
from utils.args_utils import load_args
import wandb

# TODO: Fix TEMP and add the correct target names
# TODO: Count Imbalances for each ds once
class CONCEPT_QUALITY():
  def __init__(self, model):
    self.main_args = None #args used to call the main function
    self.model = model
    self.output = None
    self.dci = None
    self.run_id = None
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
    self.metrics = {}

  def store_output(self, split = 'test'):
    logger.debug(f"Storing output for {split} split.")
    self.output = self.model.run(split)
    logger.debug(f"Output stored in CQA object.")
    return

  def save(self):
    pickle.dump(self, open(self.CQA_save_path, "wb"))
    print(self.args)
    logger.info(f"Saved to {self.CQA_save_path}")
    return

  def eval():
    pass

  def get_classification_report(self):
    y_true = self.output['labels_gt'] 
    y_pred = self.output['labels_pred'].argmax(axis=1)
    ds_name = self.model.args.dataset.split('_')[0]
    target_names = LABELS[ds_name]
    self.classification_report = cr(y_true, y_pred, target_names=target_names, output_dict=True)
    self.metrics['label_accuracy'] = self.classification_report['accuracy']
    self.metrics['label_f1'] = self.classification_report['macro avg']['f1-score']
    self.save()
    return self.classification_report

  def concept_metrics(self, threshold = 0.5):
    if self.output['concepts_gt'].dim() == 1:
      if self.output['concepts_gt'][0] == -1:
        # This means that there are no ground truth concepts
        logger.warning("No ground truth concepts found. Skipping concept metrics.")
        return None
      else:
        raise ValueError("Concepts in the wrong format.")
      
    _output = copy.deepcopy(self.output)
    if self.args.model in REQUIRES_SIGMOID:
      logger.info("Training Logistic Regression on Concepts")
      W,B = train_LR_on_concepts(_output['concepts_pred'],_output['concepts_gt'])
      _output['concepts_pred'] *= W
      _output['concepts_pred'] += B
    m = get_conceptWise_metrics(_output, self.model.args, self.main_args, threshold=threshold)

    c_aucs = compute_AUCROC_concepts(_output, self.model.args)
    self.metrcs.update({'concept_auc':c_aucs, 'avg_concept_auc':np.mean(c_aucs)})
    self.metrics.update(m)
    self.save()
    return m
  
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
    self.metrics['disentanglement'] = dci['disentanglement']
    self.metrics['completeness'] = dci['completeness']
    self.save()
    return dci

  def save_im_as_img(self, path,file_name, plot_title):
    img_path = os.path.join(path,file_name)
    heatmap(self.dci['importance_matrix'],self.model.args.dataset.split("_")[0], plot_title=plot_title, save_path=img_path)
    #save_IM_as_img(path, file_name, plot_title, self.dci['importance_matrix'])
    return 
  
  def log_metrics(self): 
    logging_metrics = {}
    log_c_accuracies = False
    for metric in METRICS:
      logger.debug(f"Logging metric: {metric}")
      #########################################
      if metric == 'concept_accuracy':  # This is because the accuracies needs to be logged separately
          x_values = range(len(self.metrics['concept_accuracy']))
          y_values = self.metrics['concept_accuracy']
          w_table = wandb.Table(columns = ["concept","concept_accuracy"])
          for x,y in zip(x_values,y_values):
              w_table.add_data(x,y)
          logging_metrics['concept_accuracy_table'] = w_table
          log_c_accuracies = True
          #for i,acc in enumerate(self.metrics['concept_accuracy']):
          #    wandb.log({f"concept_accuracy":acc, "manual_step":i})
      #########################################
      if metric in self.metrics:
        logging_metrics[metric] = self.metrics[metric]
      else:
        logger.warning(f"Missing metric: {metric}")
    if os.path.exists(os.path.join(self.main_args.load_dir,"importance_matrix.png")):
      logger.debug(f"Logging DCI image from {self.main_args.load_dir}")
      logging_metrics['DCI'] = wandb.Image(os.path.join(self.main_args.load_dir,"importance_matrix.png"))
    
    logger.debug(f"Logging {logging_metrics}")
    wandb.log(logging_metrics)  
    if log_c_accuracies:
        for i,acc in enumerate(self.metrics['concept_accuracy']):
            wandb.log({f"concept_accuracy":acc, "manual_step":i})
    return logging_metrics

def initialize_CQA(folder_path, args, split = 'test'):
  force_from_scratch = args.force
  logger.debug(f"Initializing CQA from {folder_path}")
  main_args = copy.deepcopy(args)
  # Check if CQA (Concept Quality Analysis) is already present
  if os.path.exists(folder_path + '/CQA.pkl') and not force_from_scratch:
    logger.info("CQA found. Loading CQA.")
    with open(folder_path + '/CQA.pkl', 'rb') as f:
      CQA = pickle.load(f)
    CQA.main_args = main_args
    CQA.args = load_args(args)
    CQA.save()
    try:
      logger.debug(CQA.metrics)
    except:
      CQA.metrics = {}
  else:
    main_args.run_name = os.path.basename(folder_path)
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
    CQA.args = args
    CQA.main_args = main_args
    logger.info(f"Running the model on {model.args.dataset} {split}...")
    # Run the model to get all the outputs
    CQA.store_output(split)
    CQA.save()
    
  return CQA
