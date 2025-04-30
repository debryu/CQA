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
from utils.utils import set_seed
from loguru import logger
from sklearn.metrics import classification_report as cr
from metrics.common import get_conceptWise_metrics, compute_AUCROC_concepts
from metrics.leakage import leakage_collapsing, auto_leakage
from utils.eval_models import train_LR_on_concepts
from sklearn.ensemble import RandomForestClassifier
from metrics.ois import oracle_impurity_score
from config import LABELS, METRICS, REQUIRES_SIGMOID
from utils.args_utils import load_args
import wandb
import traceback

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

  def store_output(self):
    logger.debug(f"Storing output for all splits.")
    self.output = self.model.run('test')
    self.output_train = self.model.run('train')
    self.output_val = self.model.run('val')
    logger.debug(f"Output stored in CQA object.")
    self.save()
    
    return

  def save(self):
    
    try:
      pickle.dump(self, open(self.CQA_save_path, "wb"))
    except:
      import dill
      dill_file = open(self.CQA_save_path, "wb")
      dill_file.write(dill.dumps(self))
      dill_file.close()
      logger.warning("Maybe you moved the model in another folder. Try updating the path in the args and force another analysis.")
      logger.warning("If the error is AttributeError: Can't get local object 'Backbone.__init__.<locals>.hook' then just ignore it!")
      logger.error(traceback.format_exc())
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

  
  def compute_leakage(self):
    ############################################
    ##                LEAKAGE                 ##
    ############################################
    num_labels = self.output_train['labels_pred'].shape[1]
    # (dataset:str,output_train, output_val, output_test, n_classes, args, epochs = 20, batch_size=64, device='cuda', hidden_size=1000, n_layers=3):
    lkg = auto_leakage(self.args.dataset, self.output_train, self.output_val, self.output, n_classes=num_labels, args=self.main_args)
    self.leakage = lkg
    self.metrics.update({'leakage': self.leakage})
    return lkg
    
  def compute_ois(self):
    ############################################
    ##                OIS                     ##
    ############################################
    # Randomize subset, especially for very large dataset such as celeba
    random_indexes = torch.range(0,len(self.output['concepts_pred'])-1)
    #print(len(self.output['concepts_pred']))
    #print(random_indexes[-1])
    subset_size = 6000
    # If the dataset is small, keep it 
    if len(random_indexes) < subset_size:
      subset = random_indexes
    else: # Otherwise only take 5000 random samples
        # Change this to the desired subset size
      subset = random_indexes[torch.randperm(len(random_indexes))[:subset_size]]
    subset = subset.long()
    ois = oracle_impurity_score(self.output['concepts_pred'][subset,:].numpy(), self.output['concepts_gt'][subset,:].numpy(), predictor_model_fn=RandomForestClassifier)
    self.ois = ois
    self.metrics.update({  'ois': self.ois})
    return ois
  
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
      W,B = train_LR_on_concepts(self.output_train['concepts_pred'],self.output_train['concepts_gt'])
      _output['concepts_pred'] *= W
      _output['concepts_pred'] += B

    m = get_conceptWise_metrics(_output, self.model.args, self.main_args, threshold=threshold)
    self.metrics.update(m)

    # Collapse the concepts, the domain goes from R to {0,1}. This to remove information leakage.
    _output['collapsed_concepts'] = (torch.nn.functional.sigmoid(_output['concepts_pred']) > threshold).float()
    _output['concepts_probs'] = torch.nn.functional.sigmoid(_output['concepts_pred'])
    #self.metrics.update(l)
    # Always compute auc roc on raw concept predictions, this is handled inside the function
    a = compute_AUCROC_concepts(_output, self.model.args)
    self.metrics.update(a)
    

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
  
  def dump_metrics(self):
    serializable_dict = {}
    for key,value in self.metrics.items():
      try:
        serializable_dict[key] = float(value)
      except:
        pass
    with open(os.path.join(self.model.args.load_dir, "metrics.txt"), "w") as f:
      json.dump(serializable_dict, f, indent=2)


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
  if os.path.exists(os.path.join(folder_path,'CQA.pkl')) and not force_from_scratch:
    logger.info("CQA found. Loading CQA.")
    try:
      with open(os.path.join(folder_path,'CQA.pkl'), 'rb') as f:
        CQA = pickle.load(f)
    except:
      import dill
      with open(os.path.join(folder_path,'CQA.pkl'), "wb") as dill_file:
        dill.dump(CQA, dill_file)
      #CQA = dill.load(os.path.join(folder_path,'CQA.pkl'), 'rb')
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
    CQA.store_output()
    CQA.save()
    set_seed(main_args.eval_seed)
  return CQA


def open_CQA(folder_path):
  logger.debug(f"Opening CQA from {folder_path}")
  try:
    with open(os.path.join(folder_path,'CQA.pkl'), 'rb') as f:
      CQA = pickle.load(f)
  except:
    logger.warning(f"Failed to load pickle {os.path.join(folder_path,'CQA.pkl')}")
    try:
      import dill
      print(os.listdir("./ordered_models/celeba"))
      with open(os.path.join(folder_path,'CQA.pkl'), "wb") as dill_file:
        dill.dump(CQA, dill_file)
      CQA = dill.load(os.path.join(folder_path,'CQA.pkl'), 'rb')
    except:
      logger.error(f"Failed Miserably to load {folder_path}/CQA.pkl")
      return None 
    
  return CQA