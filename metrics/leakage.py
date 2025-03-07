import torch
from loguru import logger
import torch.nn.functional as F
from models.glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from torch.utils.data import TensorDataset, random_split, Subset
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import copy

def mask_output(output, mask):
    _out = copy.deepcopy(output)
    print(_out['concepts_pred'].shape)
    print(mask.shape)
    _out['concepts_pred'] = output['concepts_pred'] * mask
    
    return _out

class FinalLayer():
    def __init__(self, in_features, out_features, lr, step_size, lam, alpha, device, reduction='mean'):
        self.linear = torch.nn.Linear(in_features, out_features).to(device)
        self.lr = lr
        self.loss_fn =  torch.nn.CrossEntropyLoss(reduction=reduction)
        self.optimizer = torch.optim.Adam(self.linear.parameters(), lr = self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=step_size, gamma=0.1)
        self.lam = lam
        self.alpha = alpha
        self.best_model = None
        self.best_loss = 1000000
    
    def train(self,loader):
        self.linear.train()
        loss_fn = self.loss_fn
        train_losses = []
        for input,label in loader:
            input = 2000*(input.to('cuda').to(torch.float32) - 0.5)
            label = label.to('cuda').long()
            output = self.linear(input).float()
            # ELASTIC LOSS
            weight, _ = list(self.linear.parameters())
            l1 = self.lam * self.alpha * weight.norm(p=1)
            l2 = 0.5 * self.lam * (1 - self.alpha) * (weight**2).sum()
            loss = loss_fn(output, label) + l1 + l2
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            train_losses.append(loss.item())
        return np.mean(train_losses)
    
    def scheduler_step(self):
        self.scheduler.step()

    def test(self,loader):
        self.linear.eval()
        loss_fn = self.loss_fn
        test_losses = []
        for input, label in loader:
            input = 2000*(input.to('cuda').to(torch.float32) - 0.5)
            label = label.to('cuda').long()
            output = self.linear(input)
            # ELASTIC LOSS
            weight, _ = list(self.linear.parameters())
            loss = loss_fn(output.float(), label)
            test_losses.append(loss.item())
        if np.mean(test_losses) <= self.best_loss:
            self.best_loss = np.mean(test_losses)
            self.best_model = copy.deepcopy(self.linear)
        return np.mean(test_losses)
    
class LeakageLayer():
    def __init__(self, in_features, out_features, n_classes, mask, lr, step_size, lam, alpha, device, reduction='mean'):
        logger.debug(f"Initializing leakage model with in_features:{in_features}, out_features:{out_features}")
        self.linear = torch.nn.Linear(in_features, out_features).to(device)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
        # Register a backward hook to enforce zero gradients where needed
        def hook_fn(grad):
            return grad * mask  # Zero out the gradients where mask is 0
        self.linear.weight.register_hook(hook_fn)
        self.lr = lr
        #self.loss_fn =  torch.nn.BCEWithLogitsLoss(reduction=reduction)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction=reduction)
        self.optimizer = torch.optim.Adam(self.linear.parameters(), lr = self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=step_size, gamma=0.1)
        self.best_model = None
        self.best_loss = 1000000
        self.n_classes = n_classes
    
    def train(self,loader):
        self.linear.train()
        loss_fn = self.loss_fn
        train_losses = []
        for input,label in loader:
            input = input.to('cuda').to(torch.float32)
            label = label.to('cuda').long()
            output = self.linear(input)
            #label = F.one_hot(label.long(), num_classes=self.n_classes).float()
            loss = loss_fn(output.float(), label)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            train_losses.append(loss.item())
            #print(self.linear.weight)
        return np.mean(train_losses)
    
    def scheduler_step(self):
        self.scheduler.step()

    def test(self,loader):
        self.linear.eval()
        loss_fn = self.loss_fn
        test_losses = []
        for input, label in loader:
            input = input.to('cuda').to(torch.float32)
            label = label.to('cuda').long()
            output = self.linear(input)
            #label = F.one_hot(label.long(), num_classes=self.n_classes).float()
            loss = loss_fn(output.float(), label)
            test_losses.append(loss.item())
        if np.mean(test_losses) <= self.best_loss:
            self.best_loss = np.mean(test_losses)
            self.best_model = copy.deepcopy(self.linear)
        return np.mean(test_losses)

def auto_leakage(output_train, output_val, output_test, n_classes, epochs = 20, batch_size=64, device='cuda', hidden_size=1000, n_layers=3):
    n_concepts = output_test['concepts_gt'].shape[1]
    # The values are in {0,1}, we want logits so need to apply the log. However this would result in -inf, inf, so we just 
    # replace them with {-1000,1000}
    #concepts_gt = output['concepts_gt'].float()
    #data = TensorDataset(concepts_gt, output['labels_gt'])
    train_dataset = TensorDataset(output_train['concepts_gt'], output_train['labels_gt'])
    val_dataset = TensorDataset(output_val['concepts_gt'], output_val['labels_gt'])
    test_dataset = TensorDataset(output_test['concepts_gt'], output_test['labels_gt'])
    # Define split ratios
    #train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.1 

    # First, split into train + (val+test)
    #train_idx, test_idx, train_labels, test_labels = train_test_split(  np.arange(len(data)), 
    #                                                                    output['labels_gt'].numpy(), 
    #                                                                    test_size=(1 - train_ratio), 
    #                                                                    stratify=output['labels_gt'].numpy(), 
    #                                                                    random_state=42
    #                                                                )
    # Split temp_idx into validation and test
    #val_idx, test_idx, _, _ = train_test_split( test_idx, 
    #                                            test_labels, 
    #                                            test_size=(test_ratio / (val_ratio + test_ratio)), 
    #                                            stratify=test_labels, 
    #                                           random_state=42
    #                                        )
    # Split into train-val-test
    #splits = [0.7,0.05]
    # Compute split sizes
    #total_size = len(data)
    #train_size = int(splits[0] * total_size)
    #val_size = int(splits[1] * total_size)
    #test_size = total_size - train_size - val_size  # Ensure all samples are used
    # Randomly split dataset
    #train_dataset = Subset(data, train_idx)
    #test_dataset = Subset(data, test_idx)
    #val_dataset = Subset(data, val_idx)
    
    #train_dataset, val_dataset, test_dataset = random_split(data, [train_size, val_size, test_size])
    
    #indexed_train_ds = IndexedTensorDataset(concepts_gt[0:train_index], output['labels_gt'].long()[0:train_index])
    #train_ds = TensorDataset(concepts_gt[0:train_index], output['labels_gt'].long()[0:train_index])
    #val_ds = TensorDataset(concepts_gt[train_index:val_index], output['labels_gt'].long()[train_index:val_index])
    #test_ds = TensorDataset(concepts_gt[val_index:], output['labels_gt'].long()[val_index:])

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    #indexed_train_loader = torch.utils.data.DataLoader(indexed_train_ds, shuffle=True, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    linear = torch.nn.Linear(n_concepts,n_classes).to(device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    metadata = {}
    metadata['max_reg'] = {}
    metadata['max_reg']['nongrouped'] = 0.000001
    logger.info("Looking for the most relevant concepts in solving the task...")

    lf = FinalLayer(in_features=n_concepts,out_features=n_classes, lr=0.001, step_size=500, lam=0.01, alpha=0.9, device = device)
    # Best result for cub after 200 epochs
    for e in tqdm(range(40)):
        train_loss = lf.train(train_loader)
        val_loss = lf.test(val_loader)
        if e % 10 == 0:
            print(f"e:{e} train:{train_loss} val:{val_loss}")
        lf.scheduler.step()


    W_g = lf.best_model.state_dict()['weight']

    predictions = []
    labels = []
    for batch in test_loader:
        inp, out = batch
        inp = inp.to(device).to(torch.float32)
        out = out.to(device).long()
        #print(labl)
        preds = lf.best_model(inp)
        preds = torch.argmax(preds.cpu(), dim=1)
        #print(preds)
        predictions.extend(preds)
        labels.extend(out.cpu().tolist())

    from sklearn.metrics import classification_report
    print(classification_report(labels,predictions))
    input("...")
    mask = torch.ones(W_g.shape).to(torch.float32)
    for i in range(n_classes):
        print(torch.topk(W_g[i], k = 10, largest=True))
        print(torch.topk(W_g[i], k = 10, largest=False))
        input("...")
        important_concepts = torch.topk(W_g[i], k = 2, largest=True)[1]
        important_negative_concepts = torch.topk(W_g[i], k = 2, largest=False)[1]
        #print('class',i,important_concepts)
        #input("...")
        mask[i][important_concepts] = 0
        mask[i][important_negative_concepts] = 0

        # Always one vs all 
        #leak = LeakageLayer(in_features=n_concepts,out_features=2, mask=mask[i].to(device), lr=0.001, step_size=500, lam=0.01, alpha=0.9, device = device)
        #for e in tqdm(range(200)):
        #    train_loss = leak.train(train_loader)
        #    val_loss = leak.test(val_loader)
        #    if e % 10 == 0:
        #        print(f"e:{e} train:{train_loss} val:{val_loss}")
            #lf.scheduler.step()

    train_dataset = TensorDataset(output_train['concepts_pred'], output_train['labels_gt'])
    val_dataset = TensorDataset(output_val['concepts_pred'], output_val['labels_gt'])
    test_dataset = TensorDataset(output_test['concepts_pred'], output_test['labels_gt'])
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    leak = LeakageLayer(in_features=n_concepts,out_features=n_classes, n_classes=n_classes, mask=mask.to(device), lr=0.001, step_size=500, lam=0.01, alpha=0.9, device = device)
    for e in tqdm(range(40)):
        train_loss = leak.train(train_loader)
        val_loss = leak.test(val_loader)
        if e % 10 == 0:
            print(f"e:{e} train:{train_loss} val:{val_loss}")
        lf.scheduler.step()

    
    predictions = []
    labels = []
    for batch in test_loader:
        inp, out = batch
        inp = inp.to(device).to(torch.float32)
        out = out.to(device).long()
        #print(labl)
        preds = leak.best_model(inp)
        preds = torch.argmax(preds.cpu(), dim=1)
        #print(preds)
        predictions.extend(preds)
        labels.extend(out.cpu().tolist())
    input("...")
    print(classification_report(labels,predictions))

    W_g = leak.best_model.state_dict()['weight']
    for i in range(n_classes):
        print(torch.topk(W_g[i], k = 2, largest=True))
        print(torch.topk(W_g[i], k = 2, largest=False))
        input("...")
    asd
def leakage_collapsing(output, n_classes, epochs = 20, batch_size=128, device='cuda', hidden_size=1000, n_layers=3):
    data = torch.utils.data.TensorDataset(output['concepts_pred'], output['labels_gt'])
    loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=batch_size)

    layers = []
    layers.append(torch.nn.Linear(output['concepts_pred'].shape[1], n_classes))
    #layers.append(torch.nn.ReLU())  # Activation function
    #for _ in range(n_layers):
    #    layers.append(torch.nn.Linear(hidden_size, hidden_size))
    #    layers.append(torch.nn.ReLU())

    #layers.append(torch.nn.Linear(hidden_size, n_classes))
    model = torch.nn.Sequential(*layers).to(device)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optmizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    best_loss_continuous = 1000000
    for e in range(epochs):
        train_losses = []
        #print(output['concepts_pred'][0])
        for batch in loader:
            _logits, _gt = batch
            _logits = _logits.to(device)
            _gt = _gt.to(device).to(torch.long)
            result = model(_logits)
            #print(result)
            #print(_gt)
            #print(result)
            #print(_gt[0])
            #print(result[0])
            loss = loss_fn(result,_gt)
            
            train_losses.append(loss.item())
            optmizer.zero_grad()
            loss.backward()
            optmizer.step()

        if np.mean(train_losses) < best_loss_continuous:
            best_loss_continuous = np.mean(train_losses)

        #logger.info(f"Epoch {e} train loss: {np.mean(train_losses)}")
    
    weights = model[0].weight
    print(best_loss_continuous)
    print(torch.topk(weights,3)[1])
    ########################################################

    torch.nn.init.zeros_(model[0].weight)
    torch.nn.init.zeros_(model[0].bias)
    data = torch.utils.data.TensorDataset(output['collapsed_concepts'], output['labels_gt'])
    loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=batch_size)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optmizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    
    best_loss_discrete = 1000000
    for e in range(epochs):
        train_losses = []
        #print(output['collapsed_concepts'][0])
        for batch in loader:
            _in, _gt = batch
            _in = _in.to(device)
            _gt = _gt.to(device).to(torch.long)
            #print(_in)
            #print(output['concepts_gt'])
            result = model(_in)
            #print(result)
            #print(_gt)
            loss = loss_fn(result,_gt)
            train_losses.append(loss.item())
            optmizer.zero_grad()
            loss.backward()
            optmizer.step()
        
        if np.mean(train_losses) < best_loss_discrete:
            best_loss_discrete = np.mean(train_losses)
        #logger.info(f"Epoch {e} train loss: {np.mean(train_losses)}")

    weights = model[0].weight
    print(best_loss_discrete)
    print(torch.topk(weights,3)[1])
    ########################################################
    print("NOT TRAINING (applying collapsed model to probs ):")

    data = torch.utils.data.TensorDataset(output['concepts_probs'], output['labels_gt'])
    loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=batch_size)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    
    train_losses = []
    #print(output['collapsed_concepts'][0])
    for batch in loader:
        _in, _gt = batch
        _in = _in.to(device)
        _gt = _gt.to(device).to(torch.long)
        #print(_in)
        #print(output['concepts_gt'])
        result = model(_in)
        #print(result)
        #print(_gt)
        loss = loss_fn(result,_gt)
        train_losses.append(loss.item())
    #logger.info(f"Epoch {e} train loss: {np.mean(train_losses)}")

    weights = model[0].weight
    print(np.mean(train_losses))  
    print(torch.topk(weights,3)[1])
    ########################################################
    print("NOT TRAINING (applying collapsed model to GT ):")

    data = torch.utils.data.TensorDataset(output['concepts_gt'].to(torch.float), output['labels_gt'])
    loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=batch_size)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    
    train_losses = []
    #print(output['collapsed_concepts'][0])
    for batch in loader:
        _in, _gt = batch
        _in = _in.to(device)
        _gt = _gt.to(device).to(torch.long)
        #print(_in)
        #print(output['concepts_gt'])
        result = model(_in)
        #print(result)
        #print(_gt)
        loss = loss_fn(result,_gt)
        train_losses.append(loss.item())
    #logger.info(f"Epoch {e} train loss: {np.mean(train_losses)}")

    weights = model[0].weight
    print(np.mean(train_losses))  
    print(torch.topk(weights,3)[1])
    ########################################################
    
    torch.nn.init.zeros_(model[0].weight)
    torch.nn.init.zeros_(model[0].bias)
    data = torch.utils.data.TensorDataset(output['concepts_probs'], output['labels_gt'])
    loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=batch_size)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optmizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    
    best_loss_discrete = 1000000
    for e in range(epochs):
        train_losses = []
        #print(output['collapsed_concepts'][0])
        for batch in loader:
            _in, _gt = batch
            _in = _in.to(device)
            _gt = _gt.to(device).to(torch.long)
            #print(_in)
            #print(output['concepts_gt'])
            result = model(_in)
            #print(result)
            #print(_gt)
            loss = loss_fn(result,_gt)
            train_losses.append(loss.item())
            optmizer.zero_grad()
            loss.backward()
            optmizer.step()
        
        if np.mean(train_losses) < best_loss_discrete:
            best_loss_discrete = np.mean(train_losses)
        #logger.info(f"Epoch {e} train loss: {np.mean(train_losses)}")

    
    weights = model[0].weight
    print(best_loss_discrete)
    print(torch.topk(weights,3)[1])
    ########################################################
    
    torch.nn.init.zeros_(model[0].weight)
    torch.nn.init.zeros_(model[0].bias)
    data = torch.utils.data.TensorDataset(output['concepts_gt'].to(torch.float), output['labels_gt'])
    loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=batch_size)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optmizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    
    best_loss_discrete = 1000000
    for e in range(epochs):
        train_losses = []
        #print(output['collapsed_concepts'][0])
        for batch in loader:
            _in, _gt = batch
            _in = _in.to(device)
            _gt = _gt.to(device).to(torch.long)
            #print(_in)
            #print(output['concepts_gt'])
            result = model(_in)
            #print(result)
            #print(_gt)
            loss = loss_fn(result,_gt)
            train_losses.append(loss.item())
            optmizer.zero_grad()
            loss.backward()
            optmizer.step()
        
        if np.mean(train_losses) < best_loss_discrete:
            best_loss_discrete = np.mean(train_losses)
        #logger.info(f"Epoch {e} train loss: {np.mean(train_losses)}")

    weights = model[0].weight
    print(best_loss_discrete)
    print(torch.topk(weights,3)[1])
    ########################################################
    print("NOT TRAINING (applying gt model to probs ):")

    data = torch.utils.data.TensorDataset(output['concepts_probs'], output['labels_gt'])
    loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=batch_size)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    
    train_losses = []
    #print(output['collapsed_concepts'][0])
    for batch in loader:
        _in, _gt = batch
        _in = _in.to(device)
        _gt = _gt.to(device).to(torch.long)
        #print(_in)
        #print(output['concepts_gt'])
        result = model(_in)
        #print(result)
        #print(_gt)
        loss = loss_fn(result,_gt)
        train_losses.append(loss.item())
    #logger.info(f"Epoch {e} train loss: {np.mean(train_losses)}")

    weights = model[0].weight
    print(np.mean(train_losses))  
    print(torch.topk(weights,3)[1])
    ########################################################
    print("NOT TRAINING (applying gt model to collapsed ):")

    data = torch.utils.data.TensorDataset(output['collapsed_concepts'], output['labels_gt'])
    loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=batch_size)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    
    train_losses = []
    #print(output['collapsed_concepts'][0])
    for batch in loader:
        _in, _gt = batch
        _in = _in.to(device)
        _gt = _gt.to(device).to(torch.long)
        #print(_in)
        #print(output['concepts_gt'])
        result = model(_in)
        #print(result)
        #print(_gt)
        loss = loss_fn(result,_gt)
        train_losses.append(loss.item())
    #logger.info(f"Epoch {e} train loss: {np.mean(train_losses)}")

    weights = model[0].weight
    print(np.mean(train_losses))  
    print(torch.topk(weights,3)[1])
    ########################################################
    
    print("RANDOM PRED")
    # Compute concept frequency
    p = output['concepts_gt'].to(torch.float).mean(dim=0)
    #p = torch.ones(p.shape)/2
    #print(p)
    #print(torch.topk(p,k=5,largest=True))
    #print(torch.topk(p,k=5,largest=False))
    n_samples = output['concepts_gt'].shape[0]
    p = p.expand(n_samples,-1)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    
    train_losses = []
    #print(output['collapsed_concepts'][0])
    for i in range(20):
        _in = torch.bernoulli(p).to(device)
        _gt = output['labels_gt'].to(device).to(torch.long)
        #print(_in)
        #print(output['concepts_gt'])
        #print(_in.shape)
        result = model(_in)
        #print(result.shape)
        #print(_gt.shape)
        #print(result)
        #print(_gt)
        loss = loss_fn(result,_gt)
        train_losses.append(loss.item())
    
    best_loss_discrete = np.mean(train_losses)
    #logger.info(f"Epoch {e} train loss: {np.mean(train_losses)}")

    print(best_loss_discrete)
    weights = model[0].weight
    print(torch.topk(weights,3)[1])
    ########################################################
    
    print("RANDOM WEIGHTS")
    train_losses = []
    for r in range(10):
        torch.nn.init.xavier_uniform_(model[0].weight)
        torch.nn.init.uniform_(model[0].bias, a=-1, b=1)
        torch.nn.init.zeros_(model[0].bias)
        data = torch.utils.data.TensorDataset(output['concepts_gt'].to(torch.float), output['labels_gt'])
        loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=batch_size)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        #print(output['collapsed_concepts'][0])
        for batch in loader:
            _in, _gt = batch
            _in = _in.to(device)
            _gt = _gt.to(device).to(torch.long)
            #print(_in)
            #print(output['concepts_gt'])
            result = model(_in)
            #print(result)
            #print(_gt)
            loss = loss_fn(result,_gt)
            train_losses.append(loss.item())
            
        
    best_loss_discrete = np.mean(train_losses)

    weights = model[0].weight
    print(best_loss_discrete)
    print(torch.topk(weights,3)[1])

    print(best_loss_continuous, best_loss_discrete)
    asd
  # First compute cross entropy loss without collapsing the concepts
  