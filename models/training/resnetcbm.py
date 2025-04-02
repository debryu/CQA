import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset
from models.glm_saga.elasticnet import IndexedTensorDataset, IndexedDataset, glm_saga
from utils.utils import log_train
from utils.lfcbm_utils import get_targets_only
from config import LABELS
from datasets import get_dataset, GenericDataset
from datasets.utils import compute_imbalance
from utils.resnetcbm_utils import get_activations_and_targets
from utils.args_utils import save_args
from models.resnetcbm import RESNETCBM
from sklearn.svm import LinearSVC

def train(args):
    # Get only the number of concepts, take the smallest ds
    args.num_classes = len(LABELS[args.dataset.split('_')[0]])
    #data = get_dataset(args.dataset, split='val', transform=None)
    data = GenericDataset(args.dataset, split='val')
    args.val_size = data.total_samples
    args.num_c = data.n_concepts
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
    args.transform = str(transform)
    data = GenericDataset(args.dataset, split='train', transform=transform)
    #data = get_dataset(args.dataset, split='train', transform=transform)
    args.train_size = len(data)

    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     #std=[0.229, 0.224, 0.225])
    
    #sampler = torch.utils.data.BatchSampler(ImbalancedDatasetSampler(data,fr), batch_size=512, drop_last=True)
    train_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_transform = model_class.get_transform(split = 'val')
    val_data = GenericDataset(args.dataset, split='val', transform=val_transform)
    #val_data = get_dataset(args.dataset, split='val', transform=val_transform)
    args.val_transform = str(val_transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')

    loss_s = []
    #fr = compute_imbalance(data)
    balancing_weight = args.balancing_weight
    if args.balanced:
        logger.debug("Balancing enabled, retrieving weights...")
        print(data.dataset)
        pos_weights = data.get_pos_weights()
        #loss_s.append(torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor(pos_weights).cuda()))
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor(pos_weights).cuda())
    loss_fn_m = torch.nn.BCEWithLogitsLoss(reduction='mean')
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
            #plt.imshow(imgs[0].permute(1, 2, 0))
            #plt.show()
            imgs = imgs.to('cuda')
            concepts = concepts.to(device='cuda', dtype=torch.float32)
            # Outputs need to be pre-sigmoid
            outputs = train_model.backbone(imgs)
            
            #outputs = torch.nn.functional.sigmoid(outputs)
            #print(outputs)
            #print(outputs.shape)
            #print(concepts.shape)
            #print(concepts)
            optimizer.zero_grad()
            loss = loss_fn(outputs, concepts)
            loss_m = loss_fn_m(outputs, concepts)
            for i in range(len(loss_s)):
                loss_m += balancing_weight*loss_s[i](outputs, concepts)
            #print(loss)
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
                outputs = train_model.backbone(imgs)
                #outputs = torch.nn.functional.sigmoid(outputs)
                #print(outputs)
                #print(concepts)
                optimizer.zero_grad()
                loss = loss_fn_m(outputs, concepts)
                val_loss.append(loss.item())
            val_loss = np.mean(val_loss)
        
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
        predictor = LinearSVC(C = 1, class_weight='balanced')
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

def train_last_layer(args):
    ''' #########################################
        ####        TRAIN LAST LAYER         ####
        #########################################
    '''
    model_class = RESNETCBM(args)
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
        indexed_train_ds = IndexedTensorDataset(train_activ_dict['concepts'], train_y)
        
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

    if args.predictor == 'svm':
        predictor = LinearSVC(C = 1, class_weight='balanced')
        predictor.fit(train_activ_dict['concepts'], train_y)
        train_acc = predictor.score(train_activ_dict['concepts'], train_y)
        test_acc = predictor.score(test_activ_dict['concepts'], test_y)
        logger.debug(f"Final layer accuracy train:{train_acc} and test: {test_acc}")
        weights = torch.tensor(predictor.coef_)
        bias = torch.tensor(predictor.intercept_)
        _out,_in= weights.shape
        if _out == 1:
            _out = 2
            w_negative = -weights
            b_negative = -bias
            weights = torch.cat((w_negative,weights), dim=0)
            bias = torch.cat((b_negative,bias), dim=0)
        W_g,b_g = weights,bias
        
    elif args.predictor == 'saga':
        # Solve the GLM path
        output_proj = glm_saga(linear, indexed_train_loader, args.glm_step_size, args.n_iters, args.glm_alpha, epsilon=1, k=1,
                        val_loader=val_loader, test_loader=test_loader, do_zero=False, metadata=metadata, n_ex=train_activ_dict['n_examples'], n_classes = len(classes))
        

        W_g = output_proj['path'][0]['weight']
        b_g = output_proj['path'][0]['bias']
    else:
        raise NotImplementedError()
            
    torch.save(W_g, os.path.join(args.save_dir, "W_g.pt"))
    torch.save(b_g, os.path.join(args.save_dir, "b_g.pt"))
    return args