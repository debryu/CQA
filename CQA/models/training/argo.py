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
from torch.utils.data import DataLoader, TensorDataset
from CQA.models.glm_saga.elasticnet import IndexedTensorDataset, IndexedDataset, glm_saga
from CQA.utils.utils import log_train
from CQA.utils.lfcbm_utils import get_targets_only
from CQA.datasets.utils import compute_imbalance
from CQA.utils.resnetcbm_utils import get_activations_and_targets
from CQA.utils.args_utils import save_args
from sklearn.svm import LinearSVC


# -----------------------------
# Loss utilities
# -----------------------------
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
def train_cbm(args, model_class, train_loader, val_loader):

    device = torch.device(args.device)  # <<< FIX >>>

    train_model = model_class.model
    train_model.to(device)              # <<< FIX >>>

    optimizer = torch.optim.Adam(train_model.parameters(), lr=0.0001)

    best_loss = float("inf")
    patience = 0

    train_model.train()
    train_model.backbone.train()

    for e in range(args.n_epochs):
        train_loss = []
        
        for imgs, concepts, labels in tqdm(train_loader, desc=f"Epoch {e}"):

            imgs = imgs.to(device, non_blocking=True)                     # <<< FIX >>>
            concepts = concepts.to(device, dtype=torch.float32,
                                    non_blocking=True)                    # <<< FIX >>>

            optimizer.zero_grad()

            output = train_model.backbone(imgs)
            if args.loss_fn == 'js':
                loss_m = compute_js(output, concepts)
            elif args.loss_fn == 'ce':
                loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
                loss_m = loss_fn(output, concepts)
            else:
                raise NotImplementedError()
            
            train_loss.append(loss_m.item())
            loss_m.backward()
            optimizer.step()

        train_loss = np.mean(train_loss)

        if e % args.val_interval == 0:
            val_loss = []

            train_model.eval()
            with torch.no_grad():                                          # <<< FIX >>>
                for imgs, concepts, labels in tqdm(val_loader,
                                                    desc=f"Validation {e}"):

                    imgs = imgs.to(device, non_blocking=True)              # <<< FIX >>>
                    concepts = concepts.to(device, dtype=torch.float32,
                                            non_blocking=True)             # <<< FIX >>>

                    output = train_model.backbone(imgs)
                    if args.loss_fn == 'js':
                        loss = compute_js(output, concepts)
                    elif args.loss_fn == 'ce':
                        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
                        loss = loss_fn(output, concepts)
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

    indexed_train_ds = IndexedTensorDataset(
        train_activ_dict["concepts"], train_y
    )
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


def prepare_cbm(args):
    # Get only the number of concepts, take the smallest ds
    args.num_classes = len(LABELS[args.dataset.split('_')[0]])
    #data = get_dataset(args.dataset, split='val', transform=None)
    data = GenericDataset(args.dataset, split='val')
    args.val_size = data.total_samples
    args.num_c = data[0][1].shape[0]
    print(args.num_c)
    if args.num_c <= 1:
        logger.warning("Bottleneck size equal to 1. Setting the bottleneck size to 128 as default.")
        args.num_c = 128
    del data

    model_class = RESNETCBM(args)
    train_model = model_class.model
    logger.info(f"Model: {train_model}")
    for name, param in train_model.named_parameters():
        logger.debug(f"{name}: requires_grad={param.requires_grad}")
    #trained_model = PretrainedResNetModel(args)
    transform = model_class.get_transform(split = 'train')
    #logger.debug(f"Train transform: {str(transform)}")
    args.transform = str(transform)
    data = GenericDataset(args.dataset, split='train')
    #data = get_dataset(args.dataset, split='train', transform=transform)
    args.train_size = len(data)

    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     #std=[0.229, 0.224, 0.225])
    
    #sampler = torch.utils.data.BatchSampler(ImbalancedDatasetSampler(data,fr), batch_size=512, drop_last=True)
    train_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    logger.debug(f"Size of train loader: {len(train_loader)}")
    val_transform = transform #model_class.get_transform(split = 'val')
    #logger.debug(f"Validation transform: {str(val_transform)}")
    val_data = GenericDataset(args.dataset, split='val')
    #val_data = get_dataset(args.dataset, split='val', transform=val_transform)
    args.val_transform = str(val_transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    logger.debug(f"Size of val loader: {len(val_loader)}")
    return train_cbm(args, model_class, train_loader, val_loader)

def train(args):
    ds = args.dataset.split('_')[0]
    dataset_train = torch.load(args.argo_train, weights_only=False, map_location='cpu')
    dataset_val = torch.load(args.argo_val, weights_only=False, map_location='cpu')
    logger.debug(f"The size of train is: {len(dataset_train)}")
    logger.debug(f"The size of val is: {len(dataset_val)}")
    args.num_classes = len(LABELS[args.dataset.split('_')[0]])
    ori_train = GenericDataset(args.dataset, split='train', transform=BaseModel.get_transform(split='train'))
    ori_val = GenericDataset(args.dataset, split='val', transform=BaseModel.get_transform(split='val'))
    ori_test = GenericDataset(args.dataset, split='test', transform=BaseModel.get_transform(split='test'))
    args.num_c = len(dataset_train[0][1])
    
    # This will not persist, as it is created runtime
    class GPDataset(torch.utils.data.Dataset):
        # Store some variables
        temp_args = args
        temp_ds = ds

        def __init__(self,**kwargs):
            split = kwargs.get('split')
            self.split = split
            self.n_concepts = args.num_c
            super().__init__()
            if split == 'train':
                self.ds = dataset_train
                self.original_data = ori_train
            elif split == 'val' or split == 'valid':
                self.ds = dataset_val
                self.original_data = ori_val
            elif split == 'test':
                self.original_data = ori_test
            else:
                raise NotImplementedError(f"Split {split} not implemented")
            
            if split != 'test':
                if len(self.ds) != len(self.original_data):
                    logger.error(f"On split {split}, len of annotated dataset = {len(self.ds)} must be the same as len of the original dataset = {len(self.original_data)}")
                assert len(self.ds) == len(self.original_data)
                    # If not, check based on the split the sizes of the datasets, and make sure the dataset and original dataset match
            
        def __len__(self):
            return len(self.original_data)
        
        def __getitem__(self, index: int):
            if self.split == 'test':
                return self.original_data[index]
            else:
                x,c,std,y = self.ds[index]
                return self.original_data[index][0], c, self.original_data[index][2]
        
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
    final_args = prepare_cbm(new_temp_args)
    vars(args).update(vars(final_args))
    
    return args

def train_last_layer(args):
    ''' #########################################
        ####        TRAIN LAST LAYER         ####
        #########################################
    '''
    pass