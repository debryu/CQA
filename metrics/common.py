from sklearn.metrics import classification_report
from loguru import logger
from config import CONCEPT_SETS
from utils.utils import get_concept_names
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

def get_conceptWise_metrics(output, model_args, main_args, theshold=0.0):
    if main_args.wandb:
        import wandb
    ds = model_args.dataset.split("_")[0]
    concept_preds = output['concepts_pred']
    concept_gt = output['concepts_gt']
    # Should be already on cpu but just in case
    concept_pred = concept_preds.cpu()
    concept_gt = concept_gt.cpu()

    # Setting concepts to 1 if the value is above the threshold, 0 otherwise
    concept_pred = (concept_pred > theshold).float()
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
        if main_args.wandb:
            wandb.log({f"concept_accuracy":cr['accuracy'], "manual_step":i})
   
    return {'avg_concept_accuracy': sum(concept_accuracies)/len(concept_accuracies), 
            'concept_accuracy':concept_accuracies, 
            'concept_classification_reports':classification_reports,
            'concept_f1':concept_f1}

def get_metrics(output, requested:list[str]):
  metrics = []
  for metric in requested:
    if metric == 'classification_report':
      metrics.append(classification_report)
  return metrics