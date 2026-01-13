from CQA.utils.load_ARGO_activations import get_activations
import torch, os
from CQA.datasets import GenericDataset
FAST_STORAGE = os.environ["FAST"]
default_results_folder = os.path.join(FAST_STORAGE,"results","fixed_models","GPs")

a = get_activations([90],['shapes3d'],['random'],['cos'],[450], results_folder=default_results_folder)
a = a[0]
subset = a['results']['training_pool']
original_dataset = GenericDataset(ds_name = a['dataset'], split = 'train')
dataset_train = torch.load(a['train'], weights_only=False, map_location='cpu')
for i in subset:
    print(dataset_train[i])
    print(original_dataset[i][1])
#dataset_val = torch.load(a['val'], weights_only=False, map_location='cpu')
#print(dataset_train[0:10])
#print(dataset_val[0:10])