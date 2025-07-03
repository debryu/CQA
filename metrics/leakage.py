import torch
from loguru import logger
import torch.nn.functional as F
from CQA.models.glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from torch.utils.data import TensorDataset, random_split, Subset
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import copy
from CQA.config import CONCEPT_SETS
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
import sklearn
from sklearn.svm import LinearSVC
from CQA.datasets import GenericDataset
from CQA.utils.utils import set_seed
import os
import pickle

lg_scaler = 100
lkg_epochs = 1
acc_metric = 'macro avg'
lam = 0.01
alpha = 0.9
def mask_output(output, mask):
    _out = copy.deepcopy(output)
    print(_out['concepts_pred'].shape)
    print(mask.shape)
    _out['concepts_pred'] = output['concepts_pred'] * mask
    
    return _out

class ConstrainedLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, device):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=False).to(device)  # Single output neuron
    
    def forward(self, x):
        w = self.linear.weight  # Shape: (1, in_features)
        W_constrained = torch.cat([w, -w], dim=0)  # Shape: (2, in_features)
        return x @ W_constrained.T  # Equivalent to a Linear transformation
    
    def get_weights(self):
        w = self.linear.weight  # Shape: (1, in_features)
        W_constrained = torch.cat([w, -w], dim=0)  # Shape: (2, in_features)
        return W_constrained
    
class FinalLayer():
    def __init__(self, in_features, out_features, lr, step_size, lam, alpha, device, svm = True, logits = True, reduction='mean'):
        # Edit #TODO
        if svm:
            self.linear = None
            clf = LinearSVC(C = 1)
        else:
            self.linear = ConstrainedLinear(in_features=in_features, out_features=1, device=device)
        
        self.lr = lr
        self.loss_fn =  torch.nn.CrossEntropyLoss(reduction=reduction)
        self.optimizer = torch.optim.Adam(self.linear.parameters(), lr = self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=step_size, gamma=0.1)
        self.patience = 0
        self.max_patience = 4
        self.lam = lam
        self.alpha = alpha
        self.best_model = None
        self.best_loss = 1000000
        self.logits = logits
        
        # Train LinearSVC model
        clf = LinearSVC(C = 1)
        import numpy as np
        all_x = []
        all_labels = []
        for batch in train_loader:
            x,y = batch
            x = x.to('cuda')
            y = y.to('cuda')
            # Store predictions and labels
            all_x.extend(x.cpu().numpy())  # Move to CPU & convert to NumPy
            all_labels.extend(y.cpu().numpy())  # Move to CPU & convert to NumPy
        clf.fit(all_x, all_labels)
    
    def train(self,loader):
        self.linear.train()
        loss_fn = self.loss_fn
        train_losses = []
        for input,label in loader:
            if not self.logits:
                input = lg_scaler*(input.to('cuda').to(torch.float32) - 0.5)
            else:
                input = input.to('cuda').to(torch.float32)
            label = label.to('cuda').long()
            output = self.linear(input).float()
            # ELASTIC LOSS
            weight, _ = list(self.linear.get_weights())
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

    def val(self,loader):
        self.linear.eval()
        loss_fn = self.loss_fn
        test_losses = []
        for input, label in loader:
            if not self.logits:
                input = lg_scaler*(input.to('cuda').to(torch.float32) - 0.5)
            else:
                input = input.to('cuda').to(torch.float32)
            label = label.to('cuda').long()
            output = self.linear(input)

            loss = loss_fn(output.float(), label)
            test_losses.append(loss.item())
        if np.mean(test_losses) <= self.best_loss:
            self.best_loss = np.mean(test_losses)
            self.best_model = copy.deepcopy(self.linear)
            self.patience += 0
        else:
            self.patience += 1
        return np.mean(test_losses)
    
    def test(self,loader):
        self.linear.eval()
        loss_fn = self.loss_fn
        test_losses = []
        for input, label in loader:
            if not self.logits:
                input = lg_scaler*(input.to('cuda').to(torch.float32) - 0.5)
            else:
                input = input.to('cuda').to(torch.float32)
            label = label.to('cuda').long()
            output = self.linear(input)
           
            loss = loss_fn(output.float(), label)
            test_losses.append(loss.item())
        return np.mean(test_losses)
    
    def stop_criterion(self):
        if self.patience > self.max_patience:
            return True
        else:
            return False
    
class LeakageLayer():
    def __init__(self, in_features, out_features, n_classes, lr, step_size, lam, alpha, device, weights = None, svm=True, mask = None, logits = True, init = None, reduction='mean'):
        
        logger.debug(f"Initializing leakage model with in_features:{in_features}, out_features:{out_features}")
        self.linear = ConstrainedLinear(in_features=in_features, out_features=1, device=device)
        if init == None:
            torch.nn.init.zeros_(self.linear.weight)
            torch.nn.init.zeros_(self.linear.bias)
        else:
            self.linear.load_state_dict({"weight":init[0], "bias":init[1]})
        # Register a backward hook to enforce zero gradients where needed
        if mask:
            def hook_fn(grad):
                return grad * mask  # Zero out the gradients where mask is 0
            self.linear.weight.register_hook(hook_fn)
        self.lr = lr
        #self.loss_fn =  torch.nn.BCEWithLogitsLoss(reduction=reduction)
        if weights is not None:
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction=reduction, weight=weights.to(device).float())
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction=reduction)
        self.patience = 0
        self.max_patience = 4
        self.optimizer = torch.optim.Adam(self.linear.parameters(), lr = self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=step_size, gamma=0.1)
        self.best_model = None
        self.best_loss = 1000000
        self.n_classes = n_classes
        self.logits = logits
        self.lam = lam
        self.alpha = alpha
    
    def train(self,loader):
        self.linear.train()
        loss_fn = self.loss_fn
        train_losses = []
        tot = len(loader)
        train_split = tot*0.7
        for i,(input,label) in enumerate(loader):
            if i>train_split:
                print(f"never seen {i}/{tot}")
                continue
            if not self.logits:
                input = lg_scaler*(input.to('cuda').to(torch.float32) - 0.5)
            else:
                input = input.to('cuda').to(torch.float32)
            label = label.to('cuda').long()
            output = self.linear(input)
            #label = F.one_hot(label.long(), num_classes=self.n_classes).float()
            weight, _ = list(self.linear.weight)
            l1 = self.lam * self.alpha * weight.norm(p=1)
            l2 = 0.5 * self.lam * (1 - self.alpha) * (weight**2).sum()
            loss = loss_fn(output.float(), label) + l1 + l2
            #loss = loss_fn(output.float(), label)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            train_losses.append(loss.item())
            #print(self.linear.weight)
        return np.mean(train_losses)
    
    def scheduler_step(self):
        self.scheduler.step()

    def val(self,loader):
        self.linear.eval()
        loss_fn = self.loss_fn
        test_losses = []
        tot = len(loader)
        train_split = tot*0.7
        for i,(input, label) in enumerate(loader):
            if i < train_split:
                print(f"never seen {i}/{tot}")
                continue
            if not self.logits:
                input = lg_scaler*(input.to('cuda').to(torch.float32) - 0.5)
            else:
                input = input.to('cuda').to(torch.float32)
            label = label.to('cuda').long()
            output = self.linear(input)
            #label = F.one_hot(label.long(), num_classes=self.n_classes).float()
            loss = loss_fn(output.float(), label)
            test_losses.append(loss.item())
        if np.mean(test_losses) <= self.best_loss:
            self.best_loss = np.mean(test_losses)
            self.best_model = copy.deepcopy(self.linear)
            self.patience += 0
        else:
            self.patience += 1
        return np.mean(test_losses)
    
    def test(self,loader):
        self.linear.eval()
        loss_fn = self.loss_fn
        test_losses = []
        for input, label in loader:
            if not self.logits:
                input = lg_scaler*(input.to('cuda').to(torch.float32) - 0.5)
            else:
                input = input.to('cuda').to(torch.float32)
            label = label.to('cuda').long()
            output = self.linear(input)
            #label = F.one_hot(label.long(), num_classes=self.n_classes).float()
            loss = loss_fn(output.float(), label)
            test_losses.append(loss.item())
        return np.mean(test_losses)
    
    def stop_criterion(self):
        if self.patience > self.max_patience:
            return True
        else:
            return False
        
    def get_weights(self):
        w = self.linear.weight  # Shape: (1, in_features)
        b = self.linear.bias
        return w,b

class LeakageSVM():
    def __init__(self, in_features, out_features, n_classes, lr, step_size, lam, alpha, device, weights = None, svm=True, mask = None, logits = True, init = None, reduction='mean', C = 1):
        logger.debug(f"Initializing leakage model with in_features:{in_features}, out_features:{out_features}")
        self.clf = LinearSVC(C = C, class_weight='balanced')
        self.best_model = None
        self.best_model = None
        self.best_loss = 1000000
        self.n_classes = n_classes
        self.logits = logits
        self.device = device
        self.weights = weights
    
    def train(self,loader, train_test_split = 0.7):
        all_x = []
        all_labels = []
        tot = len(loader)
        train_split = tot*train_test_split
        for i,batch in enumerate(loader):
            if i > train_split:
                #print(f"TRAINING never seen {i}/{tot}")
                continue
            x,y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            # Store predictions and labels
            all_x.extend(x.cpu().numpy())  # Move to CPU & convert to NumPy
            all_labels.extend(y.cpu().numpy())  # Move to CPU & convert to NumPy
        #print(all_x)
        #asd
        self.clf.fit(all_x, all_labels)

        w,b = self.get_weights()
        _out,_in= w.shape
        self.best_model = torch.nn.Linear(_in,_out).to(self.device)
        self.best_model.load_state_dict({'weight':w, 'bias':b})
        #print(self.clf.predict(all_x))
        #print("decision f")
        #print(self.clf.decision_function(all_x))
        return self.clf.score(all_x, all_labels)
    
    def scheduler_step(self):
        self.scheduler.step()

    def val(self,loader):
        all_x = []
        all_labels = []
        for batch in loader:
            x,y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            # Store predictions and labels
            all_x.extend(x.cpu().numpy())  # Move to CPU & convert to NumPy
            all_labels.extend(y.cpu().numpy())  # Move to CPU & convert to NumPy
        return self.clf.score(all_x, all_labels)
    
    def test(self,loader):
        all_x = []
        all_labels = []
        for batch in loader:
            x,y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            # Store predictions and labels
            all_x.extend(x.cpu().numpy())  # Move to CPU & convert to NumPy
            all_labels.extend(y.cpu().numpy())  # Move to CPU & convert to NumPy
        return self.clf.score(all_x, all_labels)
    
    def stop_criterion(self):
        if self.patience > self.max_patience:
            return True
        else:
            return False
        
    def get_weights(self):
        weights = torch.tensor(self.clf.coef_)
        bias = torch.tensor(self.clf.intercept_)
        _out,_in= weights.shape
        if _out == 1:
            _out = 2
            w_negative = -weights
            b_negative = -bias
            weights = torch.cat((w_negative,weights), dim=0)
            bias = torch.cat((b_negative,bias), dim=0)
        return weights,bias

def auto_leakage(dataset:str,output_train, output_val, output_test, n_classes, args, epochs = 20, batch_size=64, device='cuda', hidden_size=1000, n_layers=3):
    set_seed(args.eval_seed)
    #n_classes = output_test['labels_pred'].shape
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
    '''
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

    
    lf = FinalLayer(in_features=n_concepts,out_features=n_classes, lr=0.0001, step_size=500, lam=0.001, alpha=0.9, device = device)
    # Best result for cub after 200 epochs
    for e in tqdm(range(400)):
        train_loss = lf.train(train_loader)
        val_loss = lf.val(val_loader)
        if e % 10 == 0:
            print(f"e:{e} train:{train_loss} val:{val_loss}")
        lf.scheduler.step()
        if lf.stop_criterion():
            break


    W_g = lf.best_model.get_weights()

    predictions = []
    labels = []
    for batch in test_loader:
        inp, out = batch
        inp = lg_scaler*(inp.to('cuda').to(torch.float32) - 0.5)
        out = out.to(device).long()
        #print(labl)
        preds = lf.best_model(inp)
        preds = torch.argmax(preds.cpu(), dim=1)
        #print(preds)
        predictions.extend(preds)
        labels.extend(out.cpu().tolist())

    from sklearn.metrics import classification_report
    print(classification_report(labels,predictions))
    print(W_g)
    input("Final layer ...")
    '''

    #----------------------------
    #       ACC CURVE
    #----------------------------
    
    logger.info("Computing correlation matrix...")

    data = GenericDataset(dataset.split("_")[0],split='train')
    gt_concepts = []
    gt_labels = []
    for sample in tqdm(data, desc='Computing Correlation Matrix'):
        _,c,l = sample
        gt_concepts.append(torch.tensor(c))
        gt_concepts.append(torch.tensor(c))
        gt_labels.append(l)
    
    gt_concepts = torch.stack(gt_concepts, dim=0)
    #print(gt_concepts.shape)
    gt_concepts = torch.stack(gt_concepts, dim=0)
    #print(gt_concepts.shape)
    gt_labels = torch.tensor(gt_labels)
    #print(gt_labels.shape)
    
    #print(gt_labels.shape)
    
    # Compute the concept correlation matrix vs the label
    #print(output_train['labels_gt'][:1000])    
    subs = torch.arange(len(output_train['concepts_gt']))  # Tensor: [0, 1, 2, ..., 9]
    # Select random indices
    subs = torch.randperm(len(subs))[:2000]
    observations = torch.cat((gt_concepts[subs,:].t(),gt_labels[subs].t().unsqueeze(dim=0)), dim=0)
    #print(observations)
    #print(observations.shape)
    corr_matrix = torch.corrcoef(observations)
    import seaborn as sns

    print(corr_matrix.shape)
    
    # Save the correlation matrix

    corr_coeff = corr_matrix[-1]


    # Convert to NumPy for plotting
    corr_matrix_np = corr_matrix.numpy()
    # Plot heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_matrix_np, cmap="coolwarm", center=0, fmt=".2f")

    # Labels and title
    plt.title("Correlation Matrix Heatmap")
    plt.savefig(os.path.join(args.load_dir,"corr_mat.png"), dpi=300, bbox_inches="tight")

    #print("plot")
    
    sorted_tensor, indices = torch.sort(torch.abs(corr_coeff), descending=False)
    # Remove the last index, which is the one corresponding to the label
    indices = indices[:-1]
    print(sorted_tensor)

    '''
    with open(CONCEPT_SETS['celeba']) as f:
        concepts = f.read().split("\n")
    concepts = concepts[torch.tensor(indices).long()]
    # Plot
    x = range(len(temp))
    plt.figure(figsize=(6, 4))
    plt.plot(x, temp, marker='o', linestyle='-', label="Values")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Plot of Tensor Values")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)  # Add horizontal line at y=0
    plt.legend()
    plt.grid(True)
    # Set custom labels
    plt.xticks(x, concepts, rotation=90)
    # Show plot
    plt.savefig("plot.png", dpi=300, bbox_inches="tight")
    '''

    
    accuracies_gt = []
    accuracies_pred = []
    losses = []
    I_gt = []
    I_pred = []
    # Compute the Entropy of y
    def entropy(probabilities):
        """Compute entropy of a discrete probability distribution."""
        probabilities = probabilities.clamp(min=1e-9)  # Avoid log(0)
        return -torch.sum(probabilities * torch.log(probabilities))
    def entropy_batch(batch):
        """Compute entropy of a discrete probability distribution."""
        batch = batch.clamp(min=1e-9)  # Avoid log(0)
        return -torch.sum(batch * torch.log(batch), dim=-1)
    
    probabilites = []
    
    for i in range(n_classes):
        #print(torch.where(output_train['labels_gt']==i))
        occurr = len(torch.where(output_test['labels_gt']==i)[0])
        #print(occurr)
        total = output_test['labels_gt'].shape[0]
        probabilites.append(occurr/total)

    print(probabilites)
    H_y = entropy(torch.tensor(probabilites))
    print(H_y)
    
    last_c_model_W = None

    if dataset.split("_")[0] == 'shapes3d':
        # How concepts are distributed
        # from 0 to 9, the concept are about wall colors, etc.
        concept_shapes3d = {'wall color':9,
                          'background color':19,
                          'object color': 29,
                          'object size': 37,
                          'shape':41}
        concept_groups = list(concept_shapes3d.values())
    else: 
        concept_groups = [i for i in range(n_concepts)]
    for concept_id,start_index in tqdm(enumerate(concept_groups), desc="Computing entropy and acc"):  
        #if concept_id < 37:
        #    continue
        train_loss = None
        val_loss = None
        test_loss = None
        #if start_index <37 :
        #    continue
        #if i<=29:
        #    continue
        if dataset.split("_")[0] == 'shapes3d':
            sorted_tensor, indices = torch.sort(torch.abs(corr_coeff), descending=False)
            subset = torch.tensor(range(0,concept_groups[concept_id]+1))
        else:
            subset = indices[:start_index+1].cpu()
            #print(subset)
            #print(indices)
            #input("...")
        #print(subset)
        logger.warning(f"subset shape {output_train['concepts_pred'][subset].shape}")
        
        #print(output_train['concepts_pred'][subset].shape)
        #print(output_train['labels_gt'].shape)
        train_dataset = TensorDataset(output_train['concepts_pred'][:,subset], output_train['labels_gt'])
        val_dataset = TensorDataset(output_val['concepts_pred'][:,subset], output_val['labels_gt'])
        test_dataset = TensorDataset(output_test['concepts_pred'][:,subset], output_test['labels_gt'])
        train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
        
        weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(output_train['labels_gt']), y=output_train['labels_gt'].numpy())
        #leak = LeakageLayer(in_features=len(subset),out_features=n_classes, n_classes=n_classes, lr=0.001, step_size=80, lam=lam, alpha=alpha, weights=torch.tensor(weights), device = device, init=last_c_model_W)
        leak = LeakageSVM(C = 1, in_features=len(subset),out_features=n_classes, n_classes=n_classes, lr=0.001, step_size=80, lam=lam, alpha=alpha, weights=torch.tensor(weights), device = device, init=last_c_model_W)
        
        for e in range(lkg_epochs):
            train_loss = leak.train(test_loader)
            #val_loss = leak.val(val_loader)
            #test_loss = leak.val(test_loader)
            #print(f"e:{e} train:{train_loss} val:{val_loss} test:{test_loss}")
            #asd
            if e % 10 == 0:
                print(f"e:{e} train:{train_loss} val:{val_loss} test:{test_loss}")
            #leak.scheduler.step()

        #last_c_w, last_c_b= leak.get_weights()
        #last_c_b = torch.cat([last_c_b, torch.zeros(1).to(device)], dim=0)  # Shape: (2, in_features)
        #last_c_w = torch.cat([last_c_w, torch.zeros(2).to(device).unsqueeze(dim=1)], dim=1)
        #last_c_model_W = last_c_w, last_c_b

        
        opy_pred = torch.tensor([]).to(device)
        tot = len(test_loader)
        train_split = tot*0.7
        predictions = []
        labels = []
        for batch_idx,batch in enumerate(test_loader):
            if batch_idx < train_split:
                #print(f"TESTING: never seen {i}/{tot}")
                continue
            inp, out = batch
            inp = inp.to(device).to(torch.float32)
            out = out.to(device).long()
            #print(labl)
            #print(leak.best_model.weight)
            #print("p input:", inp)
            preds = leak.best_model(inp)
            #print("p output:", preds)
            #asd
            #print(preds)
            
            #probs = torch.nn.functional.sigmoid(leak.best_model(inp))
            #print(probs)
            #print(probs)
            #print(entropy_batch(probs))
            #opy_pred = torch.cat((opy_pred,entropy_batch(probs)), dim=0)
            #print(preds[0:10])
            #print(leak.clf.predict(inp[0:10].cpu()))
            preds = torch.argmax(preds.cpu(), dim=1)
            #print(preds.shape)
            #print(preds)
            #print(preds.view(-1))
            predictions.extend(preds)
            labels.extend(out.cpu().tolist())
            
        
        #print(predictions)
        #asd
        entr = torch.mean(opy_pred).cpu()
        #print(classification_report(labels,predictions))
        #print(labels[0:2])
        #print(labels[200:202])
        #print(predictions[0:2])
        #print(classification_report(labels,predictions,output_dict=False))
        #print(classification_report(labels,predictions, output_dict=False))
        #print("PRED")
        #input("...")
        accuracies_pred.append(classification_report(labels,predictions,output_dict=True)['macro avg']['f1-score'])
        #input("Predicted concepts ")
        #print(accuracies_pred[-1])
        #asd
        losses.append(leak.best_loss)
        I_pred.append(entr)

        ####### FOR GT concepts
        train_dataset = TensorDataset(output_train['concepts_gt'][:,subset], output_train['labels_gt'])
        val_dataset = TensorDataset(output_val['concepts_gt'][:,subset], output_val['labels_gt'])
        #print(subset)
        #print(output_test['concepts_gt'][0:10,subset])
        test_dataset = TensorDataset(output_test['concepts_gt'][:,subset], output_test['labels_gt'])
        train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
        weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(output_train['labels_gt']), y=output_train['labels_gt'].numpy())
        #leak = LeakageLayer(in_features=len(subset),out_features=n_classes, n_classes=n_classes, lr=0.001, step_size=80, lam=lam, alpha=alpha, weights=torch.tensor(weights), device = device)
        leak = LeakageSVM(C=1, in_features=len(subset),out_features=n_classes, n_classes=n_classes, lr=0.001, step_size=80, lam=lam, alpha=alpha, weights=torch.tensor(weights), device = device)
        
        for e in range(lkg_epochs):
            train_loss = leak.train(test_loader)
            #val_loss = leak.val(val_loader)
            if e % 10 == 0:
                print(f"e:{e} train:{train_loss} val:{val_loss} test:{test_loss}")
            #leak.scheduler.step()
        opy_gt = torch.tensor([]).to(device)
        tot = len(test_loader)
        train_split = tot*0.7
        labels = []
        predictions = []
        for batch_idx,batch in enumerate(test_loader):
            if batch_idx > train_split: # Shoould be <
                #print(f"TESTING: never seen {i}/{tot}")
                continue
            inp, out = batch
            inp = inp.to(device).to(torch.float32)
            out = out.to(device).long()
            #print(labl)
            #print("gt input:", inp)
            preds = leak.best_model(inp)
            asdg = inp
            #print("gt out:",preds)
            #probs = torch.nn.functional.sigmoid(leak.best_model(inp))
            #print(probs)
            #print(entropy_batch(probs))
            #opy_gt = torch.cat((opy_gt,entropy_batch(probs)), dim=0)
            #preds = torch.argmax(preds.cpu(), dim=1)
            preds = torch.argmax(preds.cpu(), dim=1)
            #print(preds)
            predictions.extend(preds)
            labels.extend(out.cpu().tolist())
       
        entr = torch.mean(opy_gt).cpu()
        #print(classification_report(labels,predictions, output_dict=False))
        #print("GT")
        #input("GT")
        accuracies_gt.append(classification_report(labels,predictions,output_dict=True)['macro avg']['f1-score'])
        #print(accuracies_gt[-1])
        losses.append(leak.best_loss)
        I_gt.append(entr)

    I_gt = torch.tensor(I_gt)
    I_gt = 1-I_gt/H_y
    I_gt = I_gt.tolist()
    I_pred = torch.tensor(I_pred)
    I_pred = 1-I_pred/H_y
    I_pred = I_pred.tolist()

    ##################################
    ##            LEAK              ##
    ##################################
    increase = []
    delta = []
    old = 0
    max_f1_gt = np.max(accuracies_gt)
    for i,f1 in enumerate(accuracies_pred):
        increment = accuracies_pred[i]-accuracies_gt[i]
        #old = accuracies_pred[i]
        if increment < 0:
            increment = 0
        normalizer = max_f1_gt-accuracies_gt[i]
        
        # When the last one concept is added and normalizer is 0, set it to one to forget about it
        if normalizer == 0.0:
            normalizer = 1.0
            
        print(normalizer)
        # If there is style leakage change the leakage computation
        if accuracies_pred[i] > max_f1_gt:
            increase.append(1 + accuracies_pred[i] - max_f1_gt)
        else:
            increase.append(increment/normalizer)
        delta.append(accuracies_pred[i]-accuracies_gt[i])
    
    print(delta[-1])
    print(increase[-1])
    # Export data
    data_to_export = {'f1_pred':accuracies_pred, 'f1_gt':accuracies_gt, 'leakage':increase, 'delta':delta}
    pickle.dump(data_to_export, open(os.path.join(args.load_dir,"leakage.data"), "wb"))
    '''
    increase = []
    old = 0
    for i,f1 in enumerate(accuracies_pred):
        increment = accuracies_pred[i]-old
        old = accuracies_pred[i]
        if increment < 0:
            increment = 0
        increase.append(increment)
    
    print(increase)
    sorted_tensor, indices_inc = torch.sort(torch.tensor(increase), descending=False)

    accuracies_pred = []
    accuracies_gt = []
    for i,start_index in tqdm(enumerate(concept_groups), desc="Computing entropy and acc"):  
        #if start_index <37 :
        #    continue
        #if i<=29:
        #    continue
        if dataset == 'shapes3d':
            sorted_tensor, indices = torch.sort(torch.abs(corr_coeff), descending=False)
            subset = torch.tensor(range(0,concept_groups[i]))
        else:
            #print(indices_inc)
            new_subset = indices_inc[:start_index+1]
            print(new_subset)
            subset = indices[new_subset].cpu()
            print(subset)
            
        #print(subset)
        logger.warning(f"subset shape {output_train['concepts_pred'][subset].shape}")
        #print(output_train['concepts_pred'][subset].shape)
        #print(output_train['labels_gt'].shape)
        train_dataset = TensorDataset(output_train['concepts_pred'][:,subset], output_train['labels_gt'])
        val_dataset = TensorDataset(output_val['concepts_pred'][:,subset], output_val['labels_gt'])
        test_dataset = TensorDataset(output_test['concepts_pred'][:,subset], output_test['labels_gt'])
        train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
        
        weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(output_train['labels_gt']), y=output_train['labels_gt'].numpy())
        #leak = LeakageLayer(in_features=len(subset),out_features=n_classes, n_classes=n_classes, lr=0.001, step_size=80, lam=lam, alpha=alpha, weights=torch.tensor(weights), device = device, init=last_c_model_W)
        leak = LeakageSVM(in_features=len(subset),out_features=n_classes, n_classes=n_classes, lr=0.001, step_size=80, lam=lam, alpha=alpha, weights=torch.tensor(weights), device = device, init=last_c_model_W)
        
        for e in range(lkg_epochs):
            train_loss = leak.train(test_loader)
            #val_loss = leak.val(val_loader)
            #test_loss = leak.val(test_loader)
            #print(f"e:{e} train:{train_loss} val:{val_loss} test:{test_loss}")
            #asd
            if e % 10 == 0:
                print(f"e:{e} train:{train_loss} val:{val_loss} test:{test_loss}")
            #leak.scheduler.step()

        #last_c_w, last_c_b= leak.get_weights()
        #last_c_b = torch.cat([last_c_b, torch.zeros(1).to(device)], dim=0)  # Shape: (2, in_features)
        #last_c_w = torch.cat([last_c_w, torch.zeros(2).to(device).unsqueeze(dim=1)], dim=1)
        #last_c_model_W = last_c_w, last_c_b

        predictions = []
        labels = []
        opy_pred = torch.tensor([]).to(device)
        tot = len(test_loader)
        train_split = tot*0.7
        for i,batch in enumerate(test_loader):
            if i < train_split:
                #print(f"TESTING: never seen {i}/{tot}")
                continue
            inp, out = batch
            inp = inp.to(device).to(torch.float32)
            out = out.to(device).long()
            #print(labl)
            #print(leak.best_model.weight)
            preds = leak.best_model(inp)
            #print(preds)
            
            #probs = torch.nn.functional.sigmoid(leak.best_model(inp))
            #print(probs)
            #print(probs)
            #print(entropy_batch(probs))
            #opy_pred = torch.cat((opy_pred,entropy_batch(probs)), dim=0)
            #print(preds[0:10])
            #print(leak.clf.predict(inp[0:10].cpu()))
            preds = (preds.cpu() > 0).long()
            #print(preds[0:10])
            #print(preds)
            predictions.extend(preds)
            labels.extend(out.cpu().tolist())
        
        
        
        entr = torch.mean(opy_pred).cpu()
        #print(classification_report(labels,predictions))
        accuracies_pred.append(classification_report(labels,predictions,output_dict=True)['macro avg']['f1-score'])
        #print(accuracies_pred[-1])
        #asd
        losses.append(leak.best_loss)
        I_pred.append(entr)

        ####### FOR GT concepts
        train_dataset = TensorDataset(output_train['concepts_gt'][:,subset], output_train['labels_gt'])
        val_dataset = TensorDataset(output_val['concepts_gt'][:,subset], output_val['labels_gt'])
        test_dataset = TensorDataset(output_test['concepts_gt'][:,subset], output_test['labels_gt'])
        train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
        weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(output_train['labels_gt']), y=output_train['labels_gt'].numpy())
        #leak = LeakageLayer(in_features=len(subset),out_features=n_classes, n_classes=n_classes, lr=0.001, step_size=80, lam=lam, alpha=alpha, weights=torch.tensor(weights), device = device)
        leak = LeakageSVM(in_features=len(subset),out_features=n_classes, n_classes=n_classes, lr=0.001, step_size=80, lam=lam, alpha=alpha, weights=torch.tensor(weights), device = device)
        
        for e in range(lkg_epochs):
            train_loss = leak.train(test_loader)
            #val_loss = leak.val(val_loader)
            if e % 10 == 0:
                print(f"e:{e} train:{train_loss} val:{val_loss} test:{test_loss}")
            #leak.scheduler.step()
        opy_gt = torch.tensor([]).to(device)
        for batch in test_loader:
            inp, out = batch
            inp = inp.to(device).to(torch.float32)
            out = out.to(device).long()
            #print(labl)
            preds = leak.best_model(inp)
            #probs = torch.nn.functional.sigmoid(leak.best_model(inp))
            #print(probs)
            #print(entropy_batch(probs))
            #opy_gt = torch.cat((opy_gt,entropy_batch(probs)), dim=0)
            #preds = torch.argmax(preds.cpu(), dim=1)
            preds = (preds.cpu() > 0).long()
            #print(preds)
            predictions.extend(preds)
            labels.extend(out.cpu().tolist())

        entr = torch.mean(opy_gt).cpu()
        #print(classification_report(labels,predictions))
        accuracies_gt.append(classification_report(labels,predictions,output_dict=True)['macro avg']['f1-score'])
        #print(accuracies_gt[-1])
        losses.append(leak.best_loss)
        I_gt.append(entr)

    I_gt = torch.tensor(I_gt)
    I_gt = 1-I_gt/H_y
    I_gt = I_gt.tolist()
    I_pred = torch.tensor(I_pred)
    I_pred = 1-I_pred/H_y
    I_pred = I_pred.tolist()
    '''
    ##################################
    ##            LEAK              ##
    ##################################
    
    with open(CONCEPT_SETS[dataset.split("_")[0]]) as f:
        concepts = f.read().split("\n")
    if dataset.split("_")[0] == 'shapes3d':
        concepts = list(concept_shapes3d.keys())
    # Plot
    x = range(len(accuracies_gt))
    plt.figure(figsize=(6, 4))
    #print(len(accuracies_pred))
    plt.plot(x, accuracies_gt, marker='o', linestyle='-', label="Acc gt", color = '#00ccff')
    plt.plot(x, accuracies_pred, marker='o', linestyle='-', label="Acc pred", color = '#ff6666')
    #plt.plot(x, losses, marker='o', linestyle='-', label="Values", color = 'red')
    #plt.plot(x, I_gt, marker='o', linestyle='-', label="1-H(y|gt)/H(y)", color = '#1D6D47')
    #plt.plot(x, I_pred, marker='o', linestyle='-', label="1-H(y|c)/H(y)", color = '#993333')
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Plot of Tensor Values")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)  # Add horizontal line at y=0
    plt.legend()
    plt.grid(True)
    #print(len(concepts))
    #print(len(indices))
    # Set custom labels
    #print(indices)
    #print(indices_inc)
    #ne_wss = indices[indices_inc]
    #print(ne_wss)
    if dataset.split("_")[0] == 'shapes3d':
        x_axis_names = concept_shapes3d.keys()
    else:
        x_axis_names = [concepts[i] for i in indices.tolist()]
    
    #x_axis_names = x_axis_names[-2:]
    plt.xticks(x, x_axis_names, rotation=90)
    if len(x_axis_names) > 80:
        plt.xticks(x, x_axis_names, rotation=90, fontsize=5)
        plt.tight_layout()
    else:
        plt.xticks(x, x_axis_names, rotation=90)
    # Show plot
    plt.savefig(os.path.join(args.load_dir, "leakage.png"), dpi=300, bbox_inches="tight")
    return np.mean(increase)
    asd
    #---------------------------------
    #       CE with all concepts
    #---------------------------------

    mask = torch.ones(W_g.shape).to(torch.float32)
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

    leak = LeakageLayer(in_features=n_concepts,out_features=n_classes, n_classes=n_classes, mask=mask.to(device), lr=0.001, step_size=500, lam=0.1, alpha=0.9, device = device)
    for e in tqdm(range(40)):
        train_loss = leak.train(train_loader)
        val_loss = leak.val(val_loader)
        if e % 10 == 0:
            print(f"e:{e} train:{train_loss} val:{val_loss}")
        leak.scheduler.step()

    
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
    
    print(classification_report(labels,predictions))
    input("All")


    #---------------------------------
    #       CE with irrelevant concepts
    #---------------------------------

    mask = torch.ones(W_g.shape).to(torch.float32)
    for i in range(n_classes):
        #print(W_g.shape)
        print(torch.topk(W_g[i], k = 3, largest=True))
        print(torch.topk(W_g[i], k = 3, largest=False))
        input(f"Class {i}")
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

    leak = LeakageLayer(in_features=n_concepts,out_features=n_classes, n_classes=n_classes, mask=mask.to(device), lr=0.001, step_size=500, lam=0.1, alpha=0.9, device = device)
    for e in tqdm(range(40)):
        train_loss = leak.train(train_loader)
        val_loss = leak.val(val_loader)
        if e % 10 == 0:
            print(f"e:{e} train:{train_loss} val:{val_loss}")
        leak.scheduler.step()

    
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
    
    print(classification_report(labels,predictions))
    input("Irrelevant")
    #------------------------------
    input("...")

    W_g = leak.best_model.state_dict()['weight']
    #W_g = lf.best_model.linear.get_weights()
    for i in range(n_classes):
        input(f"Class {i}")
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
  