from CQA.datasets import GenericDataset
import torch

for dataset in ['cub','shapes3d','dermamnist','celeba']:
    train = GenericDataset(ds_name=dataset, split = 'train')
    test = GenericDataset(ds_name=dataset, split = 'test')
    val = GenericDataset(ds_name=dataset, split = 'val')
    train_concepts = []
    for s in train:
        _,c,_l = s
        train_concepts.append(c)
    train_concepts = torch.stack(train_concepts, dim=0)
  
    torch.save(train_concepts, f"/leonardo_scratch/fast/IscrC_ARGO/train_gt_concepts_{dataset}.pt")
    
    test_concepts = []
    for s in test:
        _,c,_l = s
        test_concepts.append(c)
    test_concepts = torch.stack(test_concepts, dim=0)
    
    torch.save(test_concepts, f"/leonardo_scratch/fast/IscrC_ARGO/test_gt_concepts_{dataset}.pt")
    
    val_concepts = []
    for s in val:
        _,c,_l = s
        val_concepts.append(c)
    val_concepts = torch.stack(val_concepts, dim=0)

    torch.save(val_concepts, f"/leonardo_scratch/fast/IscrC_ARGO/val_gt_concepts_{dataset}.pt") 