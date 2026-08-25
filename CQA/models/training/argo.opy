import os
from loguru import logger
import torch
import copy
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
from CQA.config import LABELS
import numpy as np
from CQA.datasets import get_dataset, classes
from CQA.config import folder_naming_convention, ACTIVATIONS_PATH, CONCEPT_SETS, LLM_GENERATED_ANNOTATIONS
from CQA.datasets import GenericDataset
from CQA.models.base import BaseModel
from CQA.models.resnetcbm import RESNETCBM
from CQA.models.argo import ARGO
from torch.utils.data import DataLoader, TensorDataset
from CQA.models.glm_saga.elasticnet import IndexedTensorDataset, IndexedDataset, glm_saga
from CQA.utils.utils import log_train
from CQA.utils.lfcbm_utils import get_targets_only
from CQA.datasets.utils import compute_imbalance
from CQA.utils.resnetcbm_utils import get_activations_and_targets
from CQA.utils.args_utils import save_args
from sklearn.svm import LinearSVC
import math


# -----------------------------
# Loss utilities
# -----------------------------

def probability_via_heat_function(mu, var):
    den = (1 + (math.pi/8)*var)**0.5
    return torch.nn.functional.sigmoid(mu/den)

def k(mu,var, eps = 1e-06):
    std = var**0.5 + eps
    return abs(mu)/std
'''
def ce_weight(k, alpha, beta):
    return k**beta/(k**beta + alpha)

def ce_weight_cp(k, critical_point, beta):
    return ce_weight(k, critical_point**beta, beta)

def compute_weights(mu,var,critical_point,beta):
    parameter_k = k(mu,var)
    #print(probability_via_heat_function(mu, var)[0])
    #print(parameter_k[0])
    return ce_weight_cp(parameter_k, critical_point=critical_point, beta=beta)
'''
def ce_weight(k, alpha, beta, mask):
    weights = torch.ones(k.shape)
    weights[mask] = k[mask]**beta/(k[mask]**beta + alpha)
    return weights

def ce_weight_pug(p, device, mask = None, eps = 1e-06):
    max_vals = torch.minimum(p, 1 - p)
    #print(max_vals.shape)
    weights = torch.ones(p.shape).to(device) / (max_vals + eps)
    if mask is not None:
        #logger.warning(f"masking {torch.sum(mask)} entries")
        weights[mask] = 0
    return weights

def ce_weight_2(k, mask, acceptance1 = 1.5, acceptance2 = 2):
    k_flat = k.view(-1)
    k_sorted, _ = torch.sort(k_flat, descending=True)
    acceptance2 = k_sorted[len(k_flat)//(len(k_flat)*0.9)]   # 0-based index
    acceptance1 = k_sorted[len(k_flat)//70]
    weights = torch.ones(k.shape)
    not_accepted = (k < acceptance2).bool()
    rescaling = k[not_accepted]- acceptance1
    negative = (rescaling < 0)
    rescaling[negative] = 0 
    weights[not_accepted] = 1- (1-(rescaling/(acceptance2-acceptance1))**2)**0.5
    return weights

def compute_weights2(mu,var):
    parameter_k = k(mu,var)
    mask = (mu != 100).bool()
    print(torch.sum(mask))
    print(mu.shape)
    #print(probability_via_heat_function(mu, var)[0])
    #print(torch.min(parameter_k),torch.max(parameter_k))
    #print(ce_weight_cp(parameter_k, critical_point=critical_point, beta=beta, mask=mask))
    return ce_weight_2(parameter_k, mask=mask)

def ce_weight_cp(k, critical_point, beta, mask):
    return ce_weight(k, critical_point**beta, beta, mask)

def compute_weights(mu,var,critical_point,beta):
    parameter_k = k(mu,var)
    mask = (mu != 100).bool()
    #print(probability_via_heat_function(mu, var)[0])
    #print(torch.min(parameter_k),torch.max(parameter_k))
    #print(ce_weight_cp(parameter_k, critical_point=critical_point, beta=beta, mask=mask))
    return ce_weight_cp(parameter_k, critical_point=critical_point, beta=beta, mask=mask)

class JSD(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.kl = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p: torch.Tensor, q: torch.Tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))


def compute_js(logits, gt, epsilon=0.001):
    probs = torch.sigmoid(logits)
    d = 1 / (1 - 2 * epsilon)
    probs = probs / d + epsilon
    prob_dist = torch.stack([probs, 1 - probs], dim=1)
    gt = gt / d + epsilon
    gt_dist = torch.stack([gt, 1 - gt], dim=1)
    loss_fn = JSD()
    return loss_fn(prob_dist, gt_dist)


# -----------------------------
# Main CBM training
# -----------------------------
def train_cbm(args, model_class, train_loader, val_loader, alpha = 2.0):
    loss_type = 'weighted' # probability uns grad['classic', 'weighted', 'zero-one', 'heat']
    
    if loss_type == 'weighted':
        logger.debug(f"Using Weighted Loss")
        logger.debug(f"alpha={alpha}")
    device = torch.device(args.device)  
    train_model = model_class.model
    train_model.to(device)              
    optimizer = torch.optim.Adam(train_model.parameters(), lr=0.0001)

    best_loss = float("inf")
    patience = 0
    

    train_model.train()
    train_model.backbone.train()

    for e in range(args.n_epochs):
        train_loss = []
        
        for imgs, concepts, uncertainties, mus, vars, labels in tqdm(train_loader, desc=f"Epoch {e}"):
            #print(concepts[0])
            #print(probability_via_heat_function(mus,vars)[0])
            #asd
            imgs = imgs.to(device, non_blocking=True)                    
            concepts = concepts.to(device, dtype=torch.float32,
                                    non_blocking=True)                    

            optimizer.zero_grad()

            output = train_model.backbone(imgs)
            #print(output[0])
            if args.loss_fn == 'js':
                loss_m = compute_js(output, concepts)
            elif args.loss_fn == 'ce':
                loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
                loss_m = loss_fn(output, concepts)
            else:
                raise NotImplementedError()
            
            ground_truth_mask = (uncertainties == -1).bool()
            if loss_type == 'pug':
                loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
                #discretized_concepts = (concepts > 0.5).float()
                #print(discretized_concepts)
                loss_m = loss_fn(output, concepts)
                #logger.warning(f"concepts {concepts[0]}")
                mask = (k(mus,vars) < 1).bool()
                w = ce_weight_pug(probability_via_heat_function(mus,vars).to(device), device, mask = mask)
                
            if loss_type == 'heat2':
                loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
                loss_m = loss_fn(output, (probability_via_heat_function(mus,vars).to(device)>0.5).float())
                w = compute_weights2(mu=mus, var=vars)
                
                mask = (uncertainties <= 0.22).float()  # 1 if confident, 0 otherwise
                print(k(mus,vars)[0])
                print(w[0])
                print(mask[0])
                logger.warning(f"{torch.max(w-mask)} {torch.min(w-mask)}")
                loss_m = torch.mean(loss_m * w.to(args.device))
            if loss_type == 'heat':
                loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
                loss_m = loss_fn(output, (probability_via_heat_function(mus,vars).to(device)>0.5).float())
                w = compute_weights(mu=mus, var=vars, critical_point=2.576, beta=8)
                
                mask = (uncertainties <= 0.22).float()  # 1 if confident, 0 otherwise
                print(k(mus,vars)[0])
                print(w[0])
                print(mask[0])
                logger.warning(f"{torch.max(w-mask)} {torch.min(w-mask)}")
                loss_m = torch.mean(loss_m * w.to(args.device))
        
            elif loss_type == 'weighted':
                weights = ground_truth_mask
                print(uncertainties.shape)
                #weights = weights / torch.sum(weights,dim=0).detach()
                #mask = ((uncertainties >= 0.0) & (uncertainties <= 0.45)).float()
                #mask = ((uncertainties >= 0.0) & (uncertainties <= 0.45)).float()
                mask =(uncertainties >= 0.0).float()
                #print(torch.sum(weights,dim=0).detach())
                #print(torch.sum(mask,dim=0).detach())
                
                #mask = mask / (1e-02 + torch.sum(mask,dim=0).detach())
                #print(mask)
                #print(uncertainties[0:3])
                weights = weights + torch.exp(-(uncertainties * 20))*mask
                #print(weights[0:3])
                weights = weights/weights.mean()
                #print(weights.mean())
                #print(concepts[0])
                #print(uncertainties[0])
                #print(weights[0])
                assert weights.min() >= 0
                #assert weights.max() <= 1.0
                
                #weights = weights/weights.mean().detach()
                #print(weights[0])
                #print(weights.detach())
                #print(torch.sum(weights,dim=0).detach())
                logger.warning(torch.mean(weights))
                #print(loss_m)
                #loss_m = (loss_m * weights.to(args.device)).mean()
                loss_m = torch.mean(loss_m * weights.to(args.device))

            elif loss_type == 'zero-one':
                # uncertainties in [0, 0.5] or [0, 1]
                mask = (uncertainties <= 0.22).float()  # 1 if confident, 0 otherwise
                #print('unc',uncertainties)
                #print(output)
                mask = mask / torch.sum(mask,dim=0).detach()
                # Optional: normalize to keep loss scale stable
                #if mask.sum() > 0:
                #    mask = mask / mask.mean().detach()
                #print(output.cpu()*mask.cpu())
                #print(concepts.cpu()*mask.cpu())
                loss_m = (loss_m * mask.to(args.device)).mean()
            else:
                loss_m = loss_m.mean()

            loss_m.backward()
            train_loss.append(loss_m.item())
            
            optimizer.step()

        train_loss = np.mean(train_loss)

        if e % args.val_interval == 0:
            val_loss = []

            train_model.eval()
            with torch.no_grad():                                         
                for batch in tqdm(val_loader,
                                                    desc=f"Validation {e}"):
                    if len(batch) == 3:
                        imgs, concepts, labels = batch
                        imgs = imgs.to(device, non_blocking=True)              
                        concepts = concepts.to(device, dtype=torch.float32,
                                                non_blocking=True)            
                        output = train_model.backbone(imgs)
                        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
                        loss = loss_fn(output, concepts)
                    else:
                        imgs, concepts, val_unc, mus,vars, labels = batch
                        imgs = imgs.to(device, non_blocking=True)              
                        concepts = concepts.to(device, dtype=torch.float32,
                                                non_blocking=True)            
                        output = train_model.backbone(imgs)
                        if args.loss_fn == 'js':
                            loss = compute_js(output, concepts)
                        elif args.loss_fn == 'ce':
                            loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
                            loss = loss_fn(output, concepts)
                            if loss_type == 'heat':
                                loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
                                loss = loss_fn(output, probability_via_heat_function(mus,vars).to(device))
                                w = compute_weights(mu=mus, var=vars, critical_point=4, beta=8)
                                logger.warning(f"Mean weight {torch.mean(w)}")
                                loss = torch.mean(loss * w.to(args.device))
                            elif loss_type in ['zero-one']:
                                # uncertainties in [0, 0.5] or [0, 1]
                                mask = (val_unc <= 0.20).float()  # 1 if confident, 0 otherwise
                                #print('unc',uncertainties)
                                #print(output)
                                mask = mask / torch.sum(mask,dim=0).detach()
                                logger.warning(f"Mean weight {torch.mean(mask)}")
                                # Optional: normalize to keep loss scale stable
                                #if mask.sum() > 0:
                                #    mask = mask / mask.mean().detach()
                                #print(output.cpu()*mask.cpu())
                                #print(concepts.cpu()*mask.cpu())
                                loss = (loss * mask.to(args.device)).mean()
                            elif loss_type == 'weighted':
                                # uncertainties in [0, 0.5] or [0, 1]
                                mask = (val_unc <= 0.20).float()  # 1 if confident, 0 otherwise
                                #print('unc',uncertainties)
                                #print(output)
                                mask = mask / torch.sum(mask,dim=0).detach()
                                # Optional: normalize to keep loss scale stable
                                #if mask.sum() > 0:
                                #    mask = mask / mask.mean().detach()
                                #print(output.cpu()*mask.cpu())
                                #print(concepts.cpu()*mask.cpu())
                                loss = (loss * mask.to(args.device)).mean()
                            else:
                                loss = loss.mean() 
                        else:
                            raise NotImplementedError()
                    val_loss.append(loss.item())

            train_model.train()
            val_loss = np.mean(val_loss)

            if np.isnan(val_loss) or np.isnan(train_loss):
                break

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(
                    train_model.backbone.state_dict(),
                    os.path.join(args.save_dir,
                                 f"best_backbone_{args.model}.pth")
                )
                patience = 0
                logger.info(f"Best model at epoch {e}")
            else:
                patience += 1

            if patience > args.patience:
                break

            log_train(e, args, train_loss=train_loss, val_loss=val_loss)
        else:
            log_train(e, args, train_loss=train_loss)

    save_args(args)

    # -----------------------------
    # Load best backbone safely
    # -----------------------------
    state = torch.load(                                             # <<< FIX >>>
        os.path.join(args.save_dir, f"best_backbone_{args.model}.pth"),
        map_location="cpu",
        weights_only=True,
    )
    train_model.backbone.load_state_dict(state)
    train_model.backbone.to(device)                                  # <<< FIX >>>
    train_model.eval()

    # -----------------------------
    # Feature extraction
    # -----------------------------
    train_activ_dict = get_activations_and_targets(
        model_class, args.dataset, "train", args
    )
    val_activ_dict = get_activations_and_targets(
        model_class, args.dataset, "val", args
    )
    test_activ_dict = get_activations_and_targets(
        model_class, args.dataset, "test", args
    )

    train_y = torch.LongTensor(train_activ_dict["targets"])
    val_y = torch.LongTensor(val_activ_dict["targets"])
    test_y = torch.LongTensor(test_activ_dict["targets"])

    ''' OLD
    indexed_train_ds = IndexedTensorDataset(
        train_activ_dict["concepts"], train_y
    )
    val_ds = TensorDataset(val_activ_dict["concepts"], val_y)
    '''
    if loss_type == 'heat':
        indexed_train_ds = IndexedTensorDataset(probability_via_heat_function(train_activ_dict['concept_means'],train_activ_dict['concept_vars']), train_y)
        val_ds = TensorDataset(probability_via_heat_function(val_activ_dict['concept_means'],val_activ_dict['concept_vars']), val_y)
    else:
        indexed_train_ds = IndexedTensorDataset(train_activ_dict["concepts"], train_y)
        val_ds = TensorDataset(val_activ_dict["concepts"], val_y)

    test_ds = TensorDataset(test_activ_dict["concepts"], test_y)

    indexed_train_loader = DataLoader(
        indexed_train_ds,
        batch_size=args.saga_batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(val_ds,
                            batch_size=args.saga_batch_size,
                            shuffle=False)
    test_loader = DataLoader(test_ds,
                             batch_size=args.saga_batch_size,
                             shuffle=False)

    classes_ = LABELS[args.dataset.split("_")[0]]

    # -----------------------------
    # Linear model (CPU on purpose)
    # -----------------------------
    linear = torch.nn.Linear(
        train_activ_dict["concepts"].shape[1],
        len(classes_)
    )                                                                # <<< FIX >>>

    linear.weight.data.zero_()
    linear.bias.data.zero_()

    metadata = {
        "max_reg": {"nongrouped": args.lam}
    }

    if args.predictor == "saga":
        output_proj = glm_saga(
            linear,
            indexed_train_loader,
            args.glm_step_size,
            args.n_iters,
            args.glm_alpha,
            epsilon=1,
            k=1,
            val_loader=val_loader,
            test_loader=test_loader,
            do_zero=False,
            metadata=metadata,
            n_ex=train_activ_dict["n_examples"],
            n_classes=len(classes_),
        )

        W_g = output_proj["path"][0]["weight"]
        b_g = output_proj["path"][0]["bias"]

    elif args.predictor == "svm":
        predictor = LinearSVC(
            C=args.c_svm,
            class_weight="balanced",
        )
        predictor.fit(train_activ_dict["concepts"], train_y)

        weights = torch.tensor(predictor.coef_)
        bias = torch.tensor(predictor.intercept_)

        if weights.shape[0] == 1:
            weights = torch.cat([-weights, weights], dim=0)
            bias = torch.cat([-bias, bias], dim=0)

        W_g = weights
        b_g = bias

    torch.save(W_g, os.path.join(args.save_dir, "W_g.pt"))
    torch.save(b_g, os.path.join(args.save_dir, "b_g.pt"))

    return args

def train_head(args, model_class, train_loader, val_loader, concept_id):
    pass


def prepare_cbm(args, alpha):
    data = GenericDataset(args.dataset, split='train')
    '''MUs = []
    VARs = []
    for i in range(len(data)):
        imgs, conc, std, mu, var, label = data[i]
        MUs.append(mu)
        VARs.append(var)
    MUs = torch.stack(MUs, dim=0)
    VARs = torch.stack(VARs, dim=0)

    # Compute Ks
    Ks = k(MUs, VARs)
    mask = (Ks < 1).bool()
    print(torch.sum(mask, dim=0)/len(data))
    mask = (Ks < 2).bool()
    print(torch.sum(mask, dim=0)/len(data))
    mask = (Ks < 3).bool()
    print(torch.sum(mask, dim=0)/len(data))'''
    args.val_size = data.total_samples
    args.num_c = data[0][1].shape[0]
    model_class = ARGO(args)
    
    train_model = model_class.model
    '''asd
    for i in range(args.num_classes):
        train_loader = torch.utils.data.DataLoader(concepts[:,i], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
        logger.debug(f"Size of train loader: {len(train_loader)}")
        val_transform = transform 
        args.val_transform = str(val_transform)
        val_data = GenericDataset(args.dataset, split='val')
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
        logger.debug(f"Size of val loader: {len(val_loader)}")
        # Experimental debugging 
        train_head(args, model_class, train_loader, model_class.get_subset_loader('val'), concept = i)
        
    return args'''
    data = GenericDataset(args.dataset, split='train')
    
    #for d in data:
    #    _, p,s,pp,ss,_ = d
    #    print(s.shape)
    #data = get_dataset(args.dataset, split='train', transform=transform)
    args.train_size = len(data)
    
    train_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    logger.debug(f"Size of train loader: {len(train_loader)}")

    
    val_data = GenericDataset(args.dataset, split='val')
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    logger.debug(f"Size of val loader: {len(val_loader)}")
    
    return train_cbm(args, model_class, train_loader, model_class.get_subset_loader('val'), alpha = alpha)

def train(args):
    ds = args.dataset.split('_')[0]
    logger.debug("Loading ARGO activations")
    dataset_train = torch.load("/leonardo_scratch/fast/IscrC_ARGO/models/argo_train.pt", weights_only=False, map_location='cpu')
    dataset_val = torch.load("/leonardo_scratch/fast/IscrC_ARGO/models/argo_val.pt", weights_only=False, map_location='cpu')

    experiments_folder = "/leonardo_scratch/fast/IscrC_ARGO/results/GPs/GPs"
    K = 340
    #_, _, K, acq_fn, kernel, seedAndDate = args.argo_train.split("-")
    #seed = seedAndDate.split("_")[0]
    #date = seedAndDate.replace(f"{seed}_","").replace(".pt","")
    #experiment_json = f"{date}-SVGP_{ds}_{acq_fn}_{kernel}_40__SEED={seed}.json"
    import json
    with open(os.path.join(experiments_folder,"2026_02_13_18_11-MCSVGP_shapes3d_randomc_RBF_150__LOGS.json"), 'r') as jsonfile:
        results_dict = json.load(jsonfile)
    for r in results_dict:
        if r['training_pool_size'] == int(K):
            training_pool = r['training_pool']
            print(training_pool)
            break
    
    logger.debug(f"The size of train is: {len(dataset_train)}")
    logger.debug(f"The size of val is: {len(dataset_val)}")
    args.num_classes = len(LABELS[args.dataset.split('_')[0]])
    ori_train = GenericDataset(ds_name=ds, split='train', transform=ARGO.get_transform(split='train'))
    ori_val = GenericDataset(ds_name=ds, split='val', transform=ARGO.get_transform(split='val'))
    ori_test = GenericDataset(ds_name=ds, split='test', transform=ARGO.get_transform(split='test'))
    args.num_c = len(dataset_train[0][1])
    
    # This will not persist, as it is created runtime
    class GPDataset(torch.utils.data.Dataset):
        # Store some variables
        temp_args = args
        temp_ds = ds

        def __init__(self,**kwargs):
            split = kwargs.get('split')
            self.training_pool = training_pool
            
            self.split = split
            self.n_concepts = args.num_c
            super().__init__()
            
            if split == 'train':
                #Merge the train and val data, since val data is not really "ground truth" and does not give any information on the loss
                self.argo_ds = dataset_train
                #self.argo_ds = dataset_train + dataset_val
                self.original_data = ori_train
                #self.original_data = ori_train + ori_val 
            elif split == 'val' or split == 'valid':
                self.argo_ds = dataset_val
                self.original_data = ori_val
            elif split == 'test':
                self.original_data = ori_test
            else:
                raise NotImplementedError(f"Split {split} not implemented")
            
            if split != 'test':
                if len(self.argo_ds) != len(self.original_data):
                    logger.error(f"On split {split}, len of annotated dataset = {len(self.argo_ds)} must be the same as len of the original dataset = {len(self.original_data)}")
                print(len(self.argo_ds),len(self.original_data))
                assert len(self.argo_ds) == len(self.original_data)
                    # If not, check based on the split the sizes of the datasets, and make sure the dataset and original dataset match
            
        def __len__(self):
            return len(self.original_data)
        
        def __getitem__(self, index: int):
            #Experimental
            if self.split in ['test']:
                return self.original_data[index]
            else:
                if index not in self.training_pool:
                    _,c,std,mu,var,y = self.argo_ds[index]
                    return self.original_data[index][0].float(), c.float(), std.float(), torch.zeros(self.original_data[index][1].shape).float(), torch.zeros(self.original_data[index][1].shape).float(), self.original_data[index][2].long()
                else:
                    return self.original_data[index][0].float(), self.original_data[index][1].float(), -torch.ones(self.original_data[index][1].shape).float(), 100*torch.ones(self.original_data[index][1].shape).float(), torch.zeros(self.original_data[index][1].shape).float(), self.original_data[index][2].long()
        
        def get_pos_weights(self):
            raise NotImplementedError

    # Inject this dataset into the available datasets temporary
    logger.debug("Injecting dataset")
    new_temp_args = copy.deepcopy(args)
    new_temp_args.dataset = f'{args.dataset}_argo'
    
    new_temp_args.balanced = True
    classes[new_temp_args.dataset] = GPDataset
    logger.debug(f"Available datasets: {classes}")
    print(new_temp_args)
    final_args = prepare_cbm(new_temp_args, alpha=5)
    vars(args).update(vars(final_args))
    
    return args

def train_last_layer(args):
    ''' #########################################
        ####        TRAIN LAST LAYER         ####
        #########################################
    '''
    pass