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

def get_conceptWise_metrics(output, args, theshold=0.0):
    ds = args.dataset.split("_")[0]
    concept_preds = output['concepts_pred']
    concept_gt = output['concepts_gt']
    # Should be already on cpu but just in case
    concept_pred = concept_preds.cpu()
    concept_gt = concept_gt.cpu()

    # Setting concepts to 1 if the value is above the threshold, 0 otherwise
    concept_pred = (concept_pred > theshold).float()
    logger.debug(f"Number of concetps: {concept_preds.shape[1]}")
    
    print(concept_pred.T.shape)
    print(concept_gt.T.shape)

    accuracy = (concept_pred == concept_gt).sum(dim=0) / concept_gt.shape[0]
    print(accuracy)
    concept_names = get_concept_names(CONCEPT_SETS[ds])
    concept_pred_list = []
    concept_gt_list = []
    for i in range(concept_gt.shape[1]):
        concept_pred_list.append(concept_pred[:,i].numpy())
        concept_gt_list.append(concept_gt[:,i].numpy())
    
    for i in range(len(concept_pred_list)):
        print(f"Concept {i}: {concept_names[i]}")
        tn = [f"No {concept_names[i]}",f"{concept_names[i]}"]
        print(classification_report(concept_gt_list[i], concept_pred_list[i], target_names=tn))
        
    
def get_metrics(output, requested:list[str]):
  metrics = []
  for metric in requested:
    if metric == 'classification_report':
      metrics.append(classification_report)
  return metrics