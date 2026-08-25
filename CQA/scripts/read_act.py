from CQA.utils.load_ARGO_activations import get_activations
import torch, os
from CQA.datasets import GenericDataset
FAST_STORAGE = os.environ["FAST"]
default_results_folder = os.path.join(FAST_STORAGE,"results","ICML","GPs")

a = get_activations([108,109,110,111],['shapes3d','cub','celeba','dermamnist'],['random'],['cos'],[50,150,250,350], results_folder=default_results_folder)

print(a)
print(len(a))
one = a[1]

a = get_activations([108,109,110,111],['shapes3d','cub','celeba','dermamnist'],['random'],['cosb'],[50,150,250,350], results_folder=default_results_folder)
print(a)
a = a[1]

asd
original_dataset = GenericDataset(ds_name = a['dataset'], split = 'train')
dataset_train = torch.load(a['train'], weights_only=False, map_location='cpu')

c_preds = []
c_gts = []
uncs = []
for i in range(len(dataset_train)):
    _,c_pred,unc,_ = dataset_train[i]
    gt_c= original_dataset[i][1]
    c_preds.append(c_pred)
    c_gts.append(gt_c)
    uncs.append(unc)
c_preds = torch.cat(c_preds, dim=0)
c_gts = torch.cat(c_gts, dim=0)
uncs = torch.cat(uncs, dim=0)
#Change this
mask = (uncs <= 0.24).bool()
masked_concepts = c_preds[mask]
masked_gt = c_gts[mask]

print(masked_concepts)
print(masked_gt)
# Convert to numpy (sklearn expects 1D arrays)
preds_bin = (masked_concepts > 0.5).long()
gt_bin = masked_gt.long()
y_pred = preds_bin.view(-1).numpy()
y_true = gt_bin.view(-1).numpy()

from sklearn.metrics import classification_report
cr = classification_report(
        y_true,
        y_pred,
        target_names=["negative", "positive"],
        digits=4,
        zero_division=0
    )
print(cr)
#dataset_val = torch.load(a['val'], weights_only=False, map_location='cpu')
#print(dataset_train[0:10])
#print(dataset_val[0:10])