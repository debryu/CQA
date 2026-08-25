from CQA.datasets import GenericDataset
import torch

for dataset in ['celeba']:
    train = GenericDataset(ds_name=dataset, split = 'train')
    test = GenericDataset(ds_name=dataset, split = 'test')
    val = GenericDataset(ds_name=dataset, split = 'val')
    
    train_labels = []
    train_concepts = []
    for s in train:
        _,c,l = s
        train_concepts.append(c)
        train_labels.append(l)
    train_concepts = torch.stack(train_concepts, dim=0)
    train_labels = torch.stack(train_labels, dim=0)
  
    torch.save(train_concepts, f"/leonardo_scratch/fast/IscrC_ARGO/train_gt_concepts_{dataset}.pt")
    torch.save(train_labels, f"/leonardo_scratch/fast/IscrC_ARGO/train_gt_lab_{dataset}.pt")
    test_labels = []
    test_concepts = []
    for s in test:
        _,c,l = s
        test_concepts.append(c)
        test_labels.append(l)
    test_concepts = torch.stack(test_concepts, dim=0)
    test_labels = torch.stack(test_labels, dim=0)
    
    torch.save(test_labels, f"/leonardo_scratch/fast/IscrC_ARGO/test_gt_lab_{dataset}.pt")
    torch.save(test_concepts, f"/leonardo_scratch/fast/IscrC_ARGO/test_gt_concepts_{dataset}.pt")
    val_labels = []
    val_concepts = []
    for s in val:
        _,c,l = s
        val_concepts.append(c)
        val_labels.append(l)
    val_concepts = torch.stack(val_concepts, dim=0)
    val_labels = torch.stack(val_labels, dim=0)

    torch.save(val_concepts, f"/leonardo_scratch/fast/IscrC_ARGO/val_gt_concepts_{dataset}.pt") 
    torch.save(val_labels, f"/leonardo_scratch/fast/IscrC_ARGO/val_gt_lab_{dataset}.pt")