import pdb
import time
import copy
import math
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
from torch.optim import lr_scheduler
from torchvision.models.resnet import Bottleneck, BasicBlock
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset
from models.glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from utils.utils import log_train
from utils.lfcbm_utils import get_targets_only
from config import LABELS
from datasets import get_dataset
from datasets.utils import compute_imbalance
from utils.resnetcbm_utils import get_activations_and_targets

class DeepLearningModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.prefixes_of_vars_to_freeze = []
        self.optimizer_name = args.optimizer
        self.optimizer_kwargs = args.optimizer_kwargs
        self.scheduler_kwargs = args.scheduler_kwargs
        self.scheduler_type = args.scheduler_type
        

        '''
        self.num_epochs = cfg['num_epochs']
        self.optimizer_name = cfg['optimizer_name']
        self.optimizer_kwargs = cfg['optimizer_kwargs']
        self.scheduler_kwargs = cfg['scheduler_kwargs']
        self.prefixes_of_vars_to_freeze = cfg['prefixes_of_vars_to_freeze']
        '''
        self.layer_magnitudes = {}

    # ----------------- Abstract class methods to be implemented per model -----------------
    def forward(self, X):
        raise NotImplementedError()

    def forward_with_intervention(self, X, labels):
        raise NotImplementedError()

    def get_data_dict_from_dataloader(self, data):
        raise NotImplementedError()

    def loss(self, outputs, data_dict):
        raise NotImplementedError()

    def analyse_predictions(self, y_true, y_pred, info={}):
        raise NotImplementedError()

    # ----------------- Standard deep learning boilerplate train + val code -----------------
    def train_or_eval_dataset(self, dataloaders, dataset_sizes, phase, intervention=False):
        """
        Given a model, data, and a phase (train/val/test), it runs the model on the data and,
        if phase = train, we will train the model.
        """
        print('Train / Eval pass on %s dataset' % phase)
        assert phase in ['train', 'val', 'test']
        use_gpu = torch.cuda.is_available()
        if phase == 'train':
            self.train(True)  # Set model to training mode
        else:
            self.train(False)  # Set model to evaluate mode

        running_loss = 0.0
        n_batches_loaded = 0
        start_time_for_100_images = time.time()
        time_data_loading = 0
        time_forward_prop = 0
        time_backward_prop = 0
        time_update_step = 0

        # Iterate over data.
        # keep track of all labels + outputs to compute the final metrics.
        concatenated_labels = {}
        concatenated_outputs = {}
        loss_details = []
        for data in dataloaders[phase]:
            # print("We reached the beginning of the loop with %i images" % n_batches_loaded)
            t = time.time()
            n_batches_loaded += 1
            

            # Get the inputs
            data_dict = self.get_data_dict_from_dataloader(data)
            inputs = data_dict['inputs']
            labels = data_dict['labels']
            time_data_loading += time.time() - t
            t = time.time()

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward
            if intervention:
                # Under intervention, we assume some limited form of access to ground truth during test-time
                assert phase in ['val', 'test'] # Usually for evaluation purposes and not training
                outputs = self.forward_with_intervention(inputs, labels)
            else:
                outputs = self.forward(inputs)

            # Compute loss
            loss, loss_detail = self.loss(outputs, data_dict)
            loss_details.append(loss_detail)

            # Keep track of everything for correlations
            extend_dicts(concatenated_labels, labels)
            extend_dicts(concatenated_outputs, outputs)
            time_forward_prop += time.time() - t
            t = time.time()

            # Backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                time_backward_prop += time.time() - t
                t = time.time()

                self.optimizer.step()
                time_update_step += time.time() - t
                t = time.time()

            # Loss statistics
            running_loss += loss.data.item() * labels[list(labels.keys())[0]].size(0)

        epoch_loss = running_loss / dataset_sizes[phase]

        info = {
            'phase': phase,
            'dataset_size': dataset_sizes[phase],
            'epoch_loss': epoch_loss,
            'loss_details': loss_details,
        }
        metrics_for_epoch = self.analyse_predictions(concatenated_labels, concatenated_outputs, info)
        return metrics_for_epoch

    def fit(self, dataloaders, dataset_sizes):
        """
        trains the model. dataloaders + dataset sizes should have keys train, val, and test. Checked.
        """
        since = time.time()
        best_model_wts = copy.deepcopy(self.state_dict())
        best_metric_val = -np.inf
        all_metrics = {}

        for epoch in range(self.num_epochs):
            epoch_t0 = time.time()

            print('\nEpoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 60)
            metrics_for_epoch = {}
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                metrics_for_phase = self.train_or_eval_dataset(dataloaders, dataset_sizes, phase)

                # Change the learning rate.
                if phase == 'val':
                    if self.lr_scheduler_type == 'step':
                        self.scheduler.step()
                    elif self.lr_scheduler_type == 'plateau':
                        self.scheduler.step(
                            metrics_for_phase[self.metric_to_use_as_stopping_criterion])
                    else:
                        raise Exception('Not a valid scheduler type')

                    print('Current learning rate after epoch %i is' % epoch)
                    # https://github.com/pytorch/pytorch/issues/2829 get learning rate.
                    for param_group in self.optimizer.param_groups:
                        print(param_group['lr'])
                    # print(self.optimizer.state_dict())

                metrics_for_epoch.update(metrics_for_phase)
                # Deep copy the model if the validation performance is better than what we've seen so far.
                if phase == 'val' and metrics_for_phase[self.metric_to_use_as_stopping_criterion] > best_metric_val:
                    best_metric_val = metrics_for_phase[self.metric_to_use_as_stopping_criterion]
                    best_model_wts = copy.deepcopy(self.state_dict())
            all_metrics[epoch] = metrics_for_epoch

            print('Total seconds taken for epoch: %2.3f' % (time.time() - epoch_t0))
            if self.verbose['layer_magnitudes']:
                print('\n\n***\nPrinting layer magnitudes')
                self.print_layer_magnitudes(epoch)

        all_metrics['final_results'] = metrics_for_epoch
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        # Load best model weights
        self.load_state_dict(best_model_wts)
        self.train(False)  # Set model to evaluate mode
        self.state_dict = best_model_wts

        # Evaluate on test set
        all_metrics['total_seconds_to_train'] = time_elapsed
        all_metrics['test_set_results'] = self.train_or_eval_dataset(dataloaders, dataset_sizes, 'test')

        return all_metrics

    def setup_optimizers(self, optimizer_name, optimizer_kwargs, scheduler_kwargs):

        # https://github.com/pytorch/pytorch/issues/679
        if optimizer_name == 'sgd':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                       **optimizer_kwargs)
        elif optimizer_name == 'adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                        **optimizer_kwargs)
        elif optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                                    **optimizer_kwargs)
        else:
            raise Exception("Not a valid optimizer")

        
        if self.scheduler_type == 'step':
            self.scheduler = lr_scheduler.StepLR(self.optimizer,
                                                 **scheduler_kwargs)
        elif self.scheduler_type == 'plateau':
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                            **scheduler_kwargs)
        else:
            raise Exception("invalid scheduler")

    def print_layer_magnitudes(self, epoch):
        # small helper method so we can make sure the right layers are being trained.
        for name, param in self.named_parameters():
            magnitude = np.linalg.norm(param.data.cpu())
            if param not in self.layer_magnitudes:
                self.layer_magnitudes[param] = magnitude
                print("The magnitude of layer %s at epoch %i is %2.5f" % (name, epoch, magnitude))
            else:
                old_magnitude = self.layer_magnitudes[param]
                delta_magnitude = magnitude - old_magnitude
                print("The magnitude of layer %s at epoch %i is %2.5f (delta %2.5f from last epoch)" % (
                    name, epoch, magnitude, delta_magnitude))
                self.layer_magnitudes[param] = magnitude

def extend_dicts(dict1, dict2):
    if len(dict1) == 0:
        for key, val in dict2.items():
            dict1[key] = val.data.cpu().numpy()
        return

    assert set(dict1.keys()) == set(dict2.keys())
    for key, val in dict2.items():
        dict1[key] = np.concatenate([dict1[key], val.data.cpu().numpy()])
    return

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class PretrainedResNetModel(DeepLearningModel):
    def __init__(self, args, build=True):
        self.inplanes = 64
        logger.debug(args)
        super().__init__(args)
        self.dropout = args.dropout_prob
        
        self.fc_layers = [1000,args.num_c]
        self.pretrained_path = None
        self.pretrained_model_name = args.backbone
        self.pretrained_exclude_vars = None
        self.conv_layers_before_end_to_unfreeze = args.unfreeze

        # ---- Architecture based on selected model ----
        block = BasicBlock if self.pretrained_model_name in ['resnet18', 'resnet34'] else Bottleneck
        layers = {
            'resnet18': [2, 2, 2, 2],
            'resnet34': [3, 4, 6, 3],
            'resnet50': [3, 4, 6, 3],
            'resnet101': [3, 4, 23, 3],
            'resnet152': [3, 8, 36, 3],
        }[self.pretrained_model_name]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1) # nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(self.dropout, inplace=False)
        self.conv_layer_dims = { 'conv1': 64,
                                 'conv2': 128,
                                 'conv3': 256,
                                 'conv4': 512 }
        previous_layer_dims = 512 * block.expansion
        for i, layer in enumerate(self.fc_layers):
            setattr(self, 'fc' + str(i + 1), nn.Linear(previous_layer_dims, layer))
            previous_layer_dims = layer

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if build:
            self.build()

    # ----------------- Abstract class methods to be implemented per model -----------------
    def get_data_dict_from_dataloader(self, data):
        raise NotImplementedError()

    def loss(self, outputs, data_dict):
        raise NotImplementedError()

    def analyse_predictions(self, y_true, y_pred, info={}):
        raise NotImplementedError()

    # ----------------- Loading pretrained ResNet and adding fc layers -----------------
    def build(self):
        # Load pretrained resnet
        self.load_pretrained()

        # Unfreeze the pretrained weights
        self.unfreeze_conv_layers(self.conv_layers_before_end_to_unfreeze)

        # Move model to GPU
        self.cuda()

        # Setup optimizers in the DeepLearningModel class
        self.setup_optimizers(self.optimizer_name, self.optimizer_kwargs, self.scheduler_kwargs)

    def compute_cnn_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.compute_cnn_features(x)
        if self.fc_layers:
            N_layers = len(self.fc_layers)
            for i, layer in enumerate(self.fc_layers):
                fn = getattr(self, 'fc' + str(i + 1))
                x = fn(x)
                # No ReLu for last layer
                if i != N_layers - 1:
                    x = self.relu(x)
                # Cache results to get intermediate outputs
                setattr(self, 'fc%s_out' % str(i + 1), x)
        else:
            x = self.fc(x)

        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def load_pretrained(self):
        # Our own trained model
        if self.pretrained_path and len(self.pretrained_exclude_vars) > 0:
            print('[A] Loading our own pretrained model')
            own_state = self.state_dict()
            pretrained_state = torch.load(self.pretrained_path, weights_only=True)
            for name, param in pretrained_state.items():
                if any([name.startswith(var) for var in self.pretrained_exclude_vars]):
                    print('  Skipping %s' % name)
                    continue
                if isinstance(param, torch.nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                print('  Loading %s' % name)
                own_state[name].copy_(param)
            return
        elif self.pretrained_path:
            print('[B] Loading our own pretrained model')
            self.load_state_dict(torch.load(self.pretrained_path), weights_only=True)
            return

        # Public pretrained ResNet model
        N_layers = len(self.fc_layers)
        if N_layers > 1 or self.fc_layers[0] != 1000: # Check if it is default model
            logger.debug('Loading pretrained ResNet')
            incompatible, unexpected = self.load_state_dict(
                model_zoo.load_url(model_urls[self.pretrained_model_name]), strict=False)

            expected_incompatible = ['fc%d.weight' % (l + 1) for l in range(N_layers)] + \
                                    ['fc%d.bias' % (l + 1) for l in range(N_layers)]
            assert all([x in expected_incompatible for x in incompatible])
            assert all([x in ['fc.weight', 'fc.bias'] for x in unexpected])
        else:
            logger.debug('Loading pretrained ResNet')
            self.load_state_dict(model_zoo.load_url(model_urls[self.pretrained_model_name]))

    def unfreeze_conv_layers(self, conv_layers_before_end_to_unfreeze):
        param_idx = 0
        all_conv_layers = []
        for name, param in self.named_parameters():
            logger.debug("Param %i: %s" % (param_idx, name), param.data.shape)
            param_idx += 1
            conv_layer_substring = get_conv_layer_substring(name)
            if conv_layer_substring is not None and conv_layer_substring not in all_conv_layers:
                all_conv_layers.append(conv_layer_substring)
        logger.debug("All conv layers", all_conv_layers)

        # Now look conv_layers_before_end_to_unfreeze conv layers before the end, and unfreeze all layers after that.
        assert conv_layers_before_end_to_unfreeze <= len(all_conv_layers)
        if conv_layers_before_end_to_unfreeze > 0:
            conv_layers_to_unfreeze = all_conv_layers[-conv_layers_before_end_to_unfreeze:]
        else:
            conv_layers_to_unfreeze = []

        to_unfreeze = False
        for name, param in self.named_parameters():
            if not name.startswith('fc'):
                # Conv layers
                conv_layer_substring = get_conv_layer_substring(name)
                if conv_layer_substring in conv_layers_to_unfreeze:
                    to_unfreeze = True
            else:
                # Non-conv layers
                if self.prefixes_of_vars_to_freeze:
                    to_freeze = any([name.startswith(var) for var in self.prefixes_of_vars_to_freeze])
                    to_unfreeze = not to_freeze
                else:
                    to_unfreeze = True

            if to_unfreeze:
                logger.debug("Param %s is UNFROZEN" % name, param.data.shape)
            else:
                logger.debug("Param %s is FROZEN" % name, param.data.shape)
                param.requires_grad = False


# Loop over layers from beginning and freeze a couple. First we need to get the layers.
def get_conv_layer_substring(name):
    # This logic is probably more complex than it needs to be but it works.
    if name[:5] == 'layer':
        sublayer_substring = '.'.join(name.split('.')[:3])
        if 'conv' in sublayer_substring:
            return sublayer_substring
    return None


def train(args):
    logger.debug('Resampling the dataset')
    t =transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            #transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            
            #normalize,
        ])
    data = get_dataset(args.dataset, split='train', transform=t)
    if not data.has_concepts:
        args.num_c = 128
    else:
        args.num_c = data[0][1].shape[0]
    test = PretrainedResNetModel(args)

    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     #std=[0.229, 0.224, 0.225])
    
    #sampler = torch.utils.data.BatchSampler(ImbalancedDatasetSampler(data,fr), batch_size=512, drop_last=True)
    train_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_data = get_dataset(args.dataset, split='val', transform=t)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')

    loss_s = []
    fr = []
    #fr = compute_imbalance(data)
    balancing_weight = args.balancing_weight
    for ratio in fr:
        loss_s.append(torch.nn.BCEWithLogitsLoss(weight=torch.FloatTensor([ratio]).cuda()))
    loss_fn_m = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(test.parameters(), lr=0.0001)
    best_loss = 1000000
    patience = 0
    for e in range(args.n_epochs):
        train_loss = []
        for batch in tqdm(train_loader, desc=f'Epoch {e}'):
            imgs, concepts, labels = batch
            # Show the first image
            #plt.imshow(imgs[0].permute(1, 2, 0))
            #plt.show()
            imgs = imgs.to('cuda')
            concepts = concepts.to(device='cuda', dtype=torch.float32)
            outputs = test(imgs)
            #outputs = torch.nn.functional.sigmoid(outputs)
            #print(outputs)
            #print(concepts)
            optimizer.zero_grad()
            loss = loss_fn(outputs, concepts)
            loss_m = loss_fn_m(outputs, concepts)
            for i in range(len(loss_s)):
                loss_m += balancing_weight*loss_s[i](outputs[:,i], concepts[:,i])
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
                outputs = test(imgs)
                #outputs = torch.nn.functional.sigmoid(outputs)
                #print(outputs)
                #print(concepts)
                optimizer.zero_grad()
                loss = loss_fn_m(outputs, concepts)
                val_loss.append(loss.item())
            val_loss = np.mean(val_loss)
        
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(test.state_dict(), os.path.join(args.save_dir, f"best_{args.model}.pth"))
                patience = 0
                logger.info(f"Best model in epoch {e}")
            if patience > args.patience:
                break
            patience += 1
            log_train(e, args, train_loss=train_loss, val_loss=val_loss)
        else:
            log_train(e, args, train_loss=train_loss)
            

    '''#########################################
        ####        TRAIN LAST LAYER         ####
        #########################################
    '''
    # Load the best model
    model = test
    model.load_state_dict(torch.load(os.path.join(args.save_dir, f"best_{args.model}.pth"), weights_only=True))
    model.eval()

    train_activ_dict = get_activations_and_targets(model, args.dataset, 'train', args)
    val_activ_dict = get_activations_and_targets(model, args.dataset, 'val', args)
    train_targets = train_activ_dict['targets']
    val_targets = val_activ_dict['targets']

    with torch.no_grad():
        train_y = torch.LongTensor(train_targets)
        indexed_train_ds = IndexedTensorDataset(train_activ_dict['concepts'], train_y)
        val_y = torch.LongTensor(val_targets)
        val_ds = TensorDataset(train_activ_dict['concepts'],val_y)

    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.saga_batch_size, shuffle=False)

    classes = LABELS[args.dataset.split('_')[0]]
    linear = torch.nn.Linear(train_activ_dict['concepts'].shape[1],len(classes)).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    metadata = {}
    metadata['max_reg'] = {}
    metadata['max_reg']['nongrouped'] = args.lam

    # Solve the GLM path
    output_proj = glm_saga(linear, indexed_train_loader, args.glm_step_size, args.n_iters, args.glm_alpha, epsilon=1, k=1,
                    val_loader=val_loader, do_zero=False, metadata=metadata, n_ex=train_activ_dict['n_examples'], n_classes = len(classes))
    W_g = output_proj['path'][0]['weight']
    b_g = output_proj['path'][0]['bias']
            
    torch.save(W_g, os.path.join(args.save_dir, "W_g.pt"))
    torch.save(b_g, os.path.join(args.save_dir, "b_g.pt"))
    return args
