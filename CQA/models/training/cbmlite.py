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
from CQA.datasets import GenericDataset
from CQA.models.base import BaseModel
from CQA.models.resnetcbm import RESNETCBM
from torch.utils.data import DataLoader, TensorDataset
from CQA.models.glm_saga.elasticnet import IndexedTensorDataset, IndexedDataset, glm_saga
from CQA.utils.utils import log_train
from CQA.config import LABELS
from CQA.datasets import get_dataset, GenericDataset
from CQA.utils.resnetcbm_utils import get_activations_and_targets
from CQA.utils.args_utils import save_args
from CQA.models.resnetcbm import RESNETCBM
from CQA.models.cbmlite import CBMLITE
from sklearn.svm import LinearSVC

def train_cbm(args, model_class,train_loader,val_loader): 
    if 'loss_fn' in args:
        if args.loss_fn == 'mse':
            loss_fn_m = torch.nn.MSELoss()
        elif args.loss_fn == 'kl':
            loss_fn_m = torch.nn.KLDivLoss()
    else:
        loss_fn_m = torch.nn.BCEWithLogitsLoss(reduction='mean')
    
    
    train_model = model_class.model
    optimizer = torch.optim.Adam(train_model.parameters(), lr=0.0001)
    best_loss = 1000000
    patience = 0
    train_model.train()
    train_model.backbone.train()
    for e in range(args.n_epochs):
        train_loss = []
        for batch in tqdm(train_loader, desc=f'Epoch {e}'):
            imgs, concepts, labels = batch
            # Show the first image
            
            imgs = imgs.to('cuda')
            concepts = concepts.to(device='cuda', dtype=torch.float32)
            # Outputs need to be pre-sigmoid
            
            optimizer.zero_grad()
            output = train_model.backbone(imgs)
            loss_m = loss_fn_m(output, concepts)
            
            
            train_loss.append(loss_m.item())
            loss_m.backward()
            optimizer.step()
        train_loss = np.mean(train_loss)
        
        if e % args.val_interval == 0:
            val_loss = []
            for batch in tqdm(val_loader, desc=f'Validation {e}'):
                imgs, concepts, labels = batch
                
                # Show the first image
                #plt.imshow(imgs[0].permute(1, 2, 0))
                #plt.show()
                imgs = imgs.to('cuda')
                concepts = concepts.to(device='cuda', dtype=torch.float32)
                # Outputs need to be pre-sigmoid
                
                
                #outputs = torch.nn.functional.sigmoid(outputs)
                #print(outputs)
                #print(concepts)
                optimizer.zero_grad()
                output = train_model.backbone(imgs)
                loss = loss_fn_m(output, concepts)
                val_loss.append(loss.item())
            val_loss = np.mean(val_loss)

            if np.isnan(val_loss) or np.isnan(train_loss):
                break
            
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(train_model.backbone.state_dict(), os.path.join(args.save_dir, f"best_backbone_{args.model}.pth"))
                patience = 0
                logger.info(f"Best model in epoch {e}")
            if patience > args.patience:
                break
            patience += 1
            log_train(e, args, train_loss=train_loss, val_loss=val_loss)
        else:
            log_train(e, args, train_loss=train_loss)
            
    save_args(args)
    ''' #########################################
        ####        TRAIN LAST LAYER         ####
        #########################################
    '''
    # Load the best model
    model_class.model = train_model
    model_class.model.backbone.load_state_dict(torch.load(os.path.join(args.save_dir, f"best_backbone_{args.model}.pth"), weights_only=True))
    model_class.model.eval()

    train_activ_dict = get_activations_and_targets(model_class, args.dataset, 'train', args)
    val_activ_dict = get_activations_and_targets(model_class, args.dataset, 'val', args)
    test_activ_dict = get_activations_and_targets(model_class, args.dataset, 'test', args)
    train_targets = train_activ_dict['targets']
    val_targets = val_activ_dict['targets']
    test_targets = test_activ_dict['targets']
    
    with torch.no_grad():
        train_y = torch.LongTensor(train_targets)
        '''
        indexed_train_ds = IndexedTensorDataset(train_activ_dict['concepts'], train_y)
        '''
        indexed_train_ds = IndexedTensorDataset(train_activ_dict['concepts'], train_y)
        #indexed_train_ds = TensorDataset(train_activ_dict['concepts'], train_y)
        #weights = torch.tensor(data.get_label_weights()).repeat(len(indexed_train_ds),1)
        #indexed_train_ds = IndexedDataset(indexed_train_ds, sample_weight=weights)
        val_y = torch.LongTensor(val_targets)
        test_y = torch.LongTensor(test_targets)
        #print(val_activ_dict['concepts'].shape)
        #print(val_y.shape)
        val_ds = TensorDataset(val_activ_dict['concepts'],val_y)
        test_ds = TensorDataset(test_activ_dict['concepts'],test_y)


    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.saga_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.saga_batch_size, shuffle=False)

    classes = LABELS[args.dataset.split('_')[0]]
    linear = torch.nn.Linear(train_activ_dict['concepts'].shape[1],len(classes)).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    metadata = {}
    metadata['max_reg'] = {}
    metadata['max_reg']['nongrouped'] = args.lam

    if args.predictor == 'saga':
        # Solve the GLM path
        output_proj = glm_saga(linear, indexed_train_loader, args.glm_step_size, args.n_iters, args.glm_alpha, epsilon=1, k=1,
                        val_loader=val_loader, test_loader=test_loader, do_zero=False, metadata=metadata, n_ex=train_activ_dict['n_examples'], n_classes = len(classes))
                        #balancing_loss_weight = data.label_weights)
        
        W_g = output_proj['path'][0]['weight']
        b_g = output_proj['path'][0]['bias']
    elif args.predictor == 'svm':
        predictor = LinearSVC(C = args.c_svm, class_weight='balanced')
        predictor.fit(train_activ_dict['concepts'], train_y)
        train_acc = predictor.score(train_activ_dict['concepts'], train_y)
        test_acc = predictor.score(test_activ_dict['concepts'], test_y)
        logger.debug(f'Predictor accuracy train: {train_acc}, Test:{test_acc}')
        weights = torch.tensor(predictor.coef_)
        bias = torch.tensor(predictor.intercept_)
        _out,_in= weights.shape
        if _out == 1:
            _out = 2
            w_negative = -weights
            b_negative = -bias
            weights = torch.cat((w_negative,weights), dim=0)
            bias = torch.cat((b_negative,bias), dim=0)   
        W_g = weights
        b_g = bias 
        
    torch.save(W_g, os.path.join(args.save_dir, "W_g.pt"))
    torch.save(b_g, os.path.join(args.save_dir, "b_g.pt"))
    return args

def prepare_cbm(args):
    # Get only the number of concepts, take the smallest ds
    args.num_classes = len(LABELS[args.dataset.split('_')[0]])
    
    data = GenericDataset(args.dataset, split='val')
    args.val_size = data.total_samples
    
    if data.n_concepts > 1:
        args.num_c = data.n_concepts
    del data

    model_class = CBMLITE(args)
    train_model = model_class.model
    logger.info(f"Model: {train_model}")
    for name, param in train_model.named_parameters():
        logger.debug(f"{name}: requires_grad={param.requires_grad}")
    '''
    #trained_model = PretrainedResNetModel(args)
    transform = model_class.get_transform(split = 'train')
    logger.debug(f"Train transform: {str(transform)}")
    args.transform = str(transform)
   
    data = GenericDataset(args.dataset, split='train', transorm = transform)
    #data = get_dataset(args.dataset, split='train', transform=transform)
    args.train_size = len(data)

    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     #std=[0.229, 0.224, 0.225])
    
    #sampler = torch.utils.data.BatchSampler(ImbalancedDatasetSampler(data,fr), batch_size=512, drop_last=True)
    
    
    train_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True)
    val_transform = model_class.get_transform(split = 'val')
    logger.debug(f"Validation transform: {str(val_transform)}")
    val_data = GenericDataset(args.dataset, split='val', transform = val_transform)
    #val_data = get_dataset(args.dataset, split='val', transform=val_transform)
    args.val_transform = str(val_transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    '''
    train_loader = model_class.get_subset_loader('train')
    val_loader = model_class.get_subset_loader('val')
    return train_cbm(args, model_class, train_loader, val_loader)

def train(args):
    K = args.subset_size
    ds = args.dataset.split('_')[0]
    args.num_classes = len(LABELS[args.dataset.split('_')[0]])
    ori_train = GenericDataset(args.dataset, split='train', transform=CBMLITE.get_transform(split='train'))
    ori_val = GenericDataset(args.dataset, split='val', transform=CBMLITE.get_transform(split='val'))
    ori_test = GenericDataset(args.dataset, split='test', transform=CBMLITE.get_transform(split='test'))
    args.num_c = len(ori_train[0][1])
    
    
    # DEPRECATED
    '''
    # This will not persist, as it is created runtime
    class SubsetDataset(torch.utils.data.Subset):
        def __init__(self,**kwargs):
            split = kwargs.get('split')
            self.split = split
            
            if split == 'train':
                self.original_data = ori_train
            elif split == 'val' or split == 'valid':
                self.original_data = ori_val
            elif split == 'test':
                self.original_data = ori_test
            else:
                raise NotImplementedError(f"Split {split} not implemented")
            self.subset_indices = random.sample(range(len(self.original_data)), K)
            
            super().__init__(self.original_data,self.subset_indices)
        
        def get_pos_weights(self):
            raise NotImplementedError

    # Inject this dataset into the available datasets temporary
    logger.debug("Injecting dataset")
    new_temp_args = copy.deepcopy(args)
    new_temp_args.dataset = f'{args.dataset}_cbmlite'
    
    new_temp_args.balanced = True
    classes[new_temp_args.dataset] = SubsetDataset
    logger.debug(f"Available datasets: {classes}")
    print(new_temp_args)
    final_args = prepare_cbm(new_temp_args)
    '''
    final_args = prepare_cbm(args)
    vars(args).update(vars(final_args))
    
    return args

def train_last_layer(args):
    ''' #########################################
        ####        TRAIN LAST LAYER         ####
        #########################################
    '''
    pass