from sklearn.metrics import classification_report
from loguru import logger
from CQA.config import CONCEPT_SETS
from CQA.utils.utils import get_concept_names
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve, auc, roc_auc_score
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import torch
import copy
from dataclasses import dataclass
from CQA.utils.eval_models import train_LR_on_concepts

'''
Current output:
out_dict = {
        "concepts_gt": annotations,
        "concepts_pred": concepts,
        "labels_gt": labels,
        "labels_pred": preds,
        "accuracy": acc_mean / len(loader.dataset)
      }
'''

def auc_roc(X,y, model_args):
  logger.debug("auc_roc function")
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=model_args.seed)
  classifier = make_pipeline(StandardScaler(), LinearSVC(random_state=model_args.seed))
  classifier.fit(X_train, y_train)
  display = PrecisionRecallDisplay.from_estimator(
    classifier, X_test, y_test,name='LINEAR SVC', plot_chance_level=True
  )
  _ = display.ax_.set_title(f'{model_args.dataset} AUC-ROC')
  y_preds = classifier.decision_function(X_test)
  # Compute precision-recall curve
  precision, recall, _ = precision_recall_curve(y_test, y_preds)
  inv_precision, inv_recall, _ = precision_recall_curve(1-y_test, -y_preds)
  # Compute PR AUC
  pr_auc = auc(recall, precision)
  inv_pr_auc = auc(inv_recall, inv_precision)
  logger.info(f"PR AUC: {pr_auc}")
  logger.info(f"INV PR AUC: {inv_pr_auc}")
  #plt.show()
  return pr_auc

def macro_auc(concept_predictions, concept_labels):
  logger.debug("Macro-pr-auc function") 
  
  # Compute precision-recall curve
  precision, recall, _ = precision_recall_curve(concept_labels, concept_predictions)
  inv_precision, inv_recall, _ = precision_recall_curve(1-concept_labels, -concept_predictions)
  
  # Compute PR AUC
  pr_auc = auc(recall, precision)
  inv_pr_auc = auc(inv_recall, inv_precision)
  logger.info(f"PR AUC: {pr_auc}")
  logger.info(f"INV PR AUC: {inv_pr_auc}")
  #plt.show()
  return (pr_auc + inv_pr_auc)/2

@dataclass
class RocAUCResult:
    roc_auc: float

def RocAUC(concept_labels: np.ndarray, predicted_concept: np.ndarray) -> RocAUCResult:
    # Standard ROC-AUC (1 is positive)
    roc_auc = roc_auc_score(concept_labels, predicted_concept)
    return RocAUCResult(roc_auc=float(roc_auc))
  
@dataclass
class MacroAUCResult:
    macro_auc: float
    prauc0: float
    prauc1: float
    min_auc: float
    
def MacroAUC(concept_labels:np.ndarray,predicted_concept:np.ndarray) -> MacroAUCResult:
    """
    Compute macro-averaged AUC scores for predicted concepts.

    Parameters
    ----------
    concept_labels : array-like of shape (n_samples)
        Ground truth binary labels for the concept. 
    predicted_concept : array-like of shape (n_samples)
        Predicted scores or probabilities for the concept.

    Returns
    -------
    results : MacroAUCResult
        Dictionary containing evaluation metrics with the following keys:
        
        - "macro_auc" : float
            Armonic mean AUC score across all concepts.
        - "prauc0" : dict[str, float]
            AUC-PR considering the 0 of the concept as negative sample (and 1 as positive).
        - "prauc1" : int
            AUC-PR considering the 1 of the concept as negative sample (and 0 as positive).

    Examples
    --------
    >>> y_true = [[1, 0, 1], [0, 1, 0], [1, 1, 1]]
    >>> y_pred = [[0.9, 0.2, 0.8], [0.1, 0.7, 0.4], [0.8, 0.6, 0.9]]
    >>> MacroAUC(y_true, y_pred)
    {'macro_auc': 0.92,
     'per_concept_auc': {0: 0.95, 1: 0.90, 2: 0.91},
     'n_concepts': 3,
     'n_samples': 3}
    """
    precision, recall, thresholds = precision_recall_curve(concept_labels, predicted_concept)
    other_class_pr, other_class_rec, other_class_thr = precision_recall_curve(1-concept_labels, -predicted_concept)
    inverted_pr_auc = auc(other_class_rec, other_class_pr)
    pr_auc = auc(recall, precision)   
    MacroAUC = (inverted_pr_auc + pr_auc)/2
    return MacroAUCResult(macro_auc=float(MacroAUC), prauc0=float(pr_auc), prauc1=float(inverted_pr_auc), min_auc=min(float(pr_auc), float(inverted_pr_auc)))

def compute_AUCROC_concepts(output,args):
    logger.debug("Computing AUC-ROC")
    conc_pred = output['concepts_pred']
    conc_gt = output['concepts_gt']

    if not hasattr(args, 'num_c'):
      args.num_c = conc_pred.shape[1]
    
    macro_pr_aucs = []
    auc_rocs = []
    min_pr_aucs = []
    for i in tqdm(range(args.num_c), desc="Computing AUC-ROC"):
      logger.info(f"Computing AUC-ROC for concept {i}")
      X = conc_pred[:,i].detach().cpu().numpy().reshape(-1,1)
      y = conc_gt[:,i].detach().cpu().numpy()
      auc_rocs.append(auc_roc(X,y, args))
      macro_pr_aucs.append(macro_auc(X,y))
      # Convert to numpy arrays
      y_pred = conc_pred[:, i].detach().cpu().numpy().ravel()
      y_true = conc_gt[:, i].detach().cpu().numpy().ravel()
      macro_result = MacroAUC(y_true, y_pred)
      min_pr_aucs.append(macro_result.min_auc)
    auc_dict = {'avg_concept_auc':np.mean(auc_rocs), 'concept_auc': auc_rocs, 'macro_pr_auc': macro_pr_aucs, 'minauc': np.mean(min_pr_aucs), 'avg_macro_pr_auc': np.mean(macro_pr_aucs)}
    return auc_dict

def get_conceptWise_metrics(output, model_args, main_args, threshold, name = '', dict_str='concepts_pred'):
    if main_args.wandb:
        import wandb     
    ds = model_args.dataset.split("_")[0]
    concept_preds = output[dict_str]
    concept_gt = output['concepts_gt']
    # Should be already on cpu but just in case
    concept_pred = concept_preds.cpu()
    concept_gt = concept_gt.cpu()

    # Setting concepts to 1 if the value is above the threshold, 0 otherwise
    concept_pred = (torch.nn.functional.sigmoid(concept_pred) > threshold).float()
    logger.debug(f"Number of concetps: {concept_preds.shape[1]}")
    
     #print(concept_pred.T.shape)
    #print(concept_gt.T.shape)
    print(concept_pred.shape)
    accuracy = (concept_pred == concept_gt).sum(dim=0) / concept_gt.shape[0]
    #print(accuracy)
    concept_names = get_concept_names(CONCEPT_SETS[ds])
    concept_pred_list = []
    concept_gt_list = []
    for i in range(concept_gt.shape[1]):
        concept_pred_list.append(concept_pred[:,i].numpy())
        concept_gt_list.append(concept_gt[:,i].numpy())
    
    concept_accuracies = []
    concept_f1 = []
    classification_reports = []
    for i in range(len(concept_pred_list)):
        print(f"Concept {i}: {concept_names[i]}")
        tn = [f"No {concept_names[i]}",f"{concept_names[i]}"]
        cr = classification_report(concept_gt_list[i], concept_pred_list[i], target_names=tn, output_dict=True)
        classification_reports.append(cr)
        concept_f1.append(cr['macro avg']['f1-score'])  # type:ignore
        concept_accuracies.append(cr['accuracy'])       # type:ignore
        #print(classification_report(concept_gt_list[i], concept_pred_list[i], target_names=tn))
        #if main_args.wandb:
        #    print("logging",{f"concept_accuracy":cr['accuracy'], "manual_step":i})
        #   wandb.log({f"concept_accuracy":cr['accuracy'], "manual_step":i})
   
    return {f'{name}avg_concept_accuracy': sum(concept_accuracies)/len(concept_accuracies), 
            f'{name}concept_accuracy':concept_accuracies, 
            f'{name}concept_classification_reports':classification_reports,
            f'{name}avg_concept_f1': sum(concept_f1)/len(concept_f1),
            f'{name}concept_f1':concept_f1}

def get_metrics(output, requested:list[str]):
  metrics = []
  for metric in requested:
    if metric == 'classification_report':
      metrics.append(classification_report)
  return metrics

  
def compute_f1_auc(predictions, labels):
    # Create a calibration set and find the best threshold
    N = labels.shape[0]
    train_ratio = 0.2
    # Separate indices by class
    pos_idx = torch.where(labels == 1)[0]
    neg_idx = torch.where(labels == 0)[0]
    # Shuffle within each class
    pos_idx = pos_idx[torch.randperm(len(pos_idx))]
    neg_idx = neg_idx[torch.randperm(len(neg_idx))]
    # Number per class in train
    n_train_pos = max(1, int(train_ratio * len(pos_idx)))
    n_train_neg = max(1, int(train_ratio * len(neg_idx)))
    # Build splits
    train_idx = torch.cat([
        pos_idx[:n_train_pos],
        neg_idx[:n_train_neg]
    ])
    test_idx = torch.cat([
        pos_idx[n_train_pos:],
        neg_idx[n_train_neg:]
    ])
    # Shuffle final indices
    train_idx = train_idx[torch.randperm(len(train_idx))]
    test_idx = test_idx[torch.randperm(len(test_idx))]
    # Apply split
    pred_train = predictions[train_idx]
    pred_test = predictions[test_idx]

    labels_train = labels[train_idx]
    labels_test = labels[test_idx]
    
    cf = copy.deepcopy(pred_test).to('cpu')
    
    logger.info("Training Logistic Regression on All Concepts")
    
    W,B = train_LR_on_concepts(pred_train.cpu(), labels_train.cpu())
    cf *= W
    cf += B
    probs = torch.nn.functional.sigmoid(cf)      
    
    # The calibrated predictions are only on the test-test set
    preds_calibrated = (probs > 0.5).int()
    
    # The raw predictions are the entire test set
    preds_raw = (torch.nn.functional.sigmoid(predictions) > 0.5).int()
    print(predictions[0:3])
    print(preds_raw[0:3])
    print(labels[0:3])
    
    # Store the concepts from all samples in a single tensor
    concept_pred_raw = []
    concept_pred_calibrated = []
    concept_gt_raw = []
    concept_pred = []
    # Collect raw X,y
    for i in range(labels.shape[1]):
        concept_pred.append(predictions[:,i].numpy())
        concept_pred_raw.append(preds_raw[:,i].numpy())
        concept_gt_raw.append(labels[:,i].numpy()) 
        
    # Collect calibrated X,y   
    concept_gt_calibrated = []
    for i in range(labels_test.shape[1]):
        concept_pred_calibrated.append(preds_calibrated[:,i].numpy())
        concept_gt_calibrated.append(labels_test[:,i].numpy())    
    
    f1_raw = []
    f1_calibrated = []
    acc_cal = []
    rec_cal = []
    prec_cal = []
    acc_raw = []
    rec_raw = []
    prec_raw = []
    reports = []
    pr_aucs = []
    roc_aucs = []
    min_pr_aucs = []
    logger.debug("Computing concept-wise auc and f1")
    # Compute concept-wise accuracy metrics
    for i in range(labels.shape[1]):
        cr_calibrated = classification_report(concept_gt_calibrated[i], concept_pred_calibrated[i], output_dict=True)
        cr_raw = classification_report(concept_gt_raw[i], concept_pred_raw[i], output_dict=True)
        
        if i in [38,39,40,41]:
           print(classification_report(concept_gt_raw[i], concept_pred_raw[i]))
        acc_raw.append(cr_raw['accuracy'])  # type:ignore
        rec_raw.append(cr_raw['macro avg']['recall'])   # type:ignore
        prec_raw.append(cr_raw['macro avg']['precision'])   # type:ignore
        acc_cal.append(cr_calibrated['accuracy'])  # type:ignore
        rec_cal.append(cr_calibrated['macro avg']['recall'])   # type:ignore
        prec_cal.append(cr_calibrated['macro avg']['precision'])   # type:ignore
        f1_calibrated.append(cr_calibrated['macro avg']['f1-score'])  # type:ignore
        f1_raw.append(cr_raw['macro avg']['f1-score'])  # type:ignore
        reports.append((cr_calibrated, cr_raw))
        
        res1 = RocAUC(concept_gt_raw[i], concept_pred[i])
        res2 = MacroAUC(concept_gt_raw[i], concept_pred[i])
        roc_aucs.append(res1.roc_auc)
        pr_aucs.append(res2.prauc0)
        min_pr_aucs.append(res2.min_auc)
        
    print(f1_calibrated)
    res = { 'f1_cal': np.mean(f1_calibrated),
            'f1_raw': np.mean(f1_raw),
            'all_f1_cal': f1_calibrated,
            'all_f1_raw': f1_raw,
            'acc_cal': np.mean(acc_cal),
            'rec_cal': np.mean(rec_cal),
            'prec_cal':np.mean(prec_cal),
            'acc_raw': np.mean(acc_raw),
            'rec_raw': np.mean(rec_raw),
            'prec_raw':np.mean(prec_raw),
            'reports':reports,
            'min_pr_auc':np.mean(min_pr_aucs),
            'pr_auc':np.mean(pr_aucs),
            'roc_auc':np.mean(roc_aucs),
            'all_roc_aucs': roc_aucs,
            'all_pr_aucs': pr_aucs,
            }
    logger.info(f"Evaluation f1: raw={res['f1_raw']} cal={res['f1_cal']}")
    return res
