from sklearn.metrics import classification_report
from loguru import logger
from config import CONCEPT_SETS
from utils.utils import get_concept_names
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve, auc
from tqdm import tqdm
from matplotlib import pyplot as plt
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

def auc_roc(X,y, model_args, concept_name=None):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=model_args.seed)
  classifier = make_pipeline(StandardScaler(), LinearSVC(random_state=model_args.seed))
  classifier.fit(X_train, y_train)
  display = PrecisionRecallDisplay.from_estimator(
    classifier, X_test, y_test,name='LINEAR SVC', plot_chance_level=True
  )
  if concept_name is not None:
    _ = display.ax_.set_title(f'{model_args.dataset} {concept_name} AUC-ROC')
  else:
    _ = display.ax_.set_title(f'{model_args.dataset} AUC-ROC')
  y_preds = classifier.decision_function(X_test)
  # Compute precision-recall curve
  precision, recall, _ = precision_recall_curve(y_test, y_preds)

  # Compute PR AUC
  pr_auc = auc(recall, precision)
  logger.info(f"PR AUC: {pr_auc}")
  plt.show()
  
  return pr_auc

def compute_AUCROC_concepts(output,args):
    if 'pre-sigmoid_concepts' in output.keys():
      conc_pred = output['pre-sigmoid_concepts']
    else:
      conc_pred = output['concepts_pred']
    conc_gt = output['concepts_gt']

    concepts_auc = []
    for i in tqdm(range(args.num_c), desc="Computing AUC-ROC"):
      logger.info(f"Computing AUC-ROC for concept {i}")
      X = conc_pred[:,i].detach().cpu().numpy().reshape(-1,1)
      y = conc_gt[:,i].detach().cpu().numpy()
      concept_name = get_concept_names(CONCEPT_SETS[args.dataset.split("_")[0]])[i]
      cauc = auc_roc(X,y, args, concept_name=concept_name)
      concepts_auc.append(cauc)
    return concepts_auc
    

def get_conceptWise_metrics(output, model_args, main_args, threshold=0.0):
    if main_args.wandb:
        import wandb
    ds = model_args.dataset.split("_")[0]
    concept_preds = output['concepts_pred']
    concept_gt = output['concepts_gt']
    # Should be already on cpu but just in case
    concept_pred = concept_preds.cpu()
    concept_gt = concept_gt.cpu()

    # Setting concepts to 1 if the value is above the threshold, 0 otherwise
    concept_pred = (concept_pred > threshold).float()
    logger.debug(f"Number of concetps: {concept_preds.shape[1]}")
    
     #print(concept_pred.T.shape)
    #print(concept_gt.T.shape)

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
        concept_f1.append(cr['macro avg']['f1-score'])
        concept_accuracies.append(cr['accuracy'])
        #print(classification_report(concept_gt_list[i], concept_pred_list[i], target_names=tn))
        #if main_args.wandb:
        #    print("logging",{f"concept_accuracy":cr['accuracy'], "manual_step":i})
        #   wandb.log({f"concept_accuracy":cr['accuracy'], "manual_step":i})
   
    return {'avg_concept_accuracy': sum(concept_accuracies)/len(concept_accuracies), 
            'concept_accuracy':concept_accuracies, 
            'concept_classification_reports':classification_reports,
            'avg_concept_f1': sum(concept_f1)/len(concept_f1),
            'concept_f1':concept_f1}

def get_metrics(output, requested:list[str]):
  metrics = []
  for metric in requested:
    if metric == 'classification_report':
      metrics.append(classification_report)
  return metrics