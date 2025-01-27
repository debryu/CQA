import argparse
import json
from config import CONCEPT_SETS, SAVED_MODELS_FOLDER, ACTIVATIONS_PATH

'''
DEFINE HERE THE PARSE FN FOR EACH TRAINING MODEL

The name needs to be as follows: parse_<model_name>_args(parser, args)
e.g. "def parse_lfcbm_args(parser, args)"

'''
# TODO: Move all glm args to a separate file
# Which also mean to change the name of the variables in the different training models

def parse_glm_args(parser, args):
  parser.add_argument("-glm_alpha", type=float, default=0.99, help="Alpha parameter for glm_saga")
  parser.add_argument("-glm_step_size", type=float, default=0.1, help="Step size for glm_saga")
  parser.add_argument("-n_iters", type=int, default=2000, help="How many iterations to run the final layer solver for")
  parser.add_argument("-lam", type=float, default=0.0007, help="Sparsity regularization parameter, higher->more sparse")
  
  return parser

def parse_lfcbm_args(parser, args):
  model = args.model
  ds = args.dataset.split("_")[0]
  parser.add_argument("-concept_set", type=str, default=f"{CONCEPT_SETS[ds]}", 
                      help="Path to the concept set name file to use")
  parser.add_argument("-backbone", type=str, default="clip_RN50", help="Which pretrained model to use as backbone")
  parser.add_argument("-clip_name", type=str, default="ViT-B/16", help="Which CLIP model to use")
  parser.add_argument("-device", type=str, default="cuda", help="Which device to use")
  parser.add_argument("-batch_size", type=int, default=258, help="Batch size used when saving model/CLIP activations")
  parser.add_argument("-saga_batch_size", type=int, default=256, help="Batch size used when fitting final layer")
  parser.add_argument("-proj_batch_size", type=int, default=50000, help="Batch size to use when learning projection layer")
  parser.add_argument("-feature_layer", type=str, default='layer4', 
                      help="Which layer to collect activations from. Should be the name of second to last layer in the model")
  parser.add_argument("-activation_dir", type=str, default='shared', help="save location for backbone and CLIP activations")
  parser.add_argument("-save_dir", type=str, default=SAVED_MODELS_FOLDER[model], help="where to save trained models")
  parser.add_argument("-clip_cutoff", type=float, default=0.25, help="concepts with smaller top5 clip activation will be deleted")
  parser.add_argument("-proj_steps", type=int, default=1000, help="how many steps to train the projection layer for")
  parser.add_argument("-interpretability_cutoff", type=float, default=0.45, help="concepts with smaller similarity to target concept will be deleted")
  parser.add_argument("-print", action='store_true', help="Print all concepts being deleted in this stage")
  parse_glm_args(parser, args)
  return parser

def parse_labo_args(parser, args):
    model = args.model
    ds = args.dataset.split("_")[0]
    parser.add_argument("-concept_set", type=str, default=f"{CONCEPT_SETS[ds]}", 
                        help="Path to the concept set name file to use")
    parser.add_argument("-backbone", type=str, default="clip_RN50", help="Which pretrained model to use as backbone")
    parser.add_argument("-clip_name", type=str, default="ViT-B/16", help="Which CLIP model to use")
    parser.add_argument("-device", type=str, default="cuda", help="Which device to use")
    parser.add_argument("-batch_size", type=int, default=258, help="Batch size used when saving model/CLIP activations")
    parser.add_argument("-saga_batch_size", type=int, default=256, help="Batch size used when fitting final layer")
    parser.add_argument("-proj_batch_size", type=int, default=50000, help="Batch size to use when learning projection layer")
    parser.add_argument("-feature_layer", type=str, default='layer4', 
                        help="Which layer to collect activations from. Should be the name of second to last layer in the model")
    parser.add_argument("-activation_dir", type=str, default='shared', help="save location for backbone and CLIP activations")
    parser.add_argument("-save_dir", type=str, default=SAVED_MODELS_FOLDER[model], help="where to save trained models")
    parser.add_argument("-clip_cutoff", type=float, default=0.25, help="concepts with smaller top5 clip activation will be deleted")
    parser.add_argument("-proj_steps", type=int, default=1000, help="how many steps to train the projection layer for")
    parser.add_argument("-interpretability_cutoff", type=float, default=0.45, help="concepts with smaller similarity to target concept will be deleted")
    parser.add_argument("-lam", type=float, default=0.0007, help="Sparsity regularization parameter, higher->more sparse")
    parser.add_argument("-n_iters", type=int, default=1000, help="How many iterations to run the final layer solver for")
    parser.add_argument("-print", action='store_true', help="Print all concepts being deleted in this stage")
    parse_glm_args(parser, args)
    return parser

def parse_vlgcbm_args(parser, args):
  ds = args.dataset.split("_")[0]
  parser.add_argument("-wandb",action="store_true",help="activate wandb")
  parser.add_argument("-mock",action="store_true",help="Mock training for debugging purposes",)
  parser.add_argument("-concept_set", type=str, default=CONCEPT_SETS[ds], help="path to concept set name")
  parser.add_argument("-filter_set", type=str, default=None, help="path to concept set name")
  parser.add_argument("-val_split", type=float, default=0.1, help="Validation split fraction")
  parser.add_argument("-backbone",type=str,default="clip_RN50",help="Which pretrained model to use as backbone",)
  parser.add_argument("-feature_layer",type=str,default="layer4",help="Which layer to collect activations from. Should be the name of second to last layer in the model",)
  parser.add_argument("-use_clip_penultimate",action="store_true",help="Whether to use the penultimate layer of the CLIP backbone",)
  parser.add_argument("-skip_concept_filter",action="store_true",help="Whether to skip filtering concepts",)
  parser.add_argument("-annotation_dir",type=str,default="./data/VLG_annotations",help="where annotations are saved",)
  parser.add_argument("-device", type=str, default="cuda", help="Which device to use")
  parser.add_argument("-num_workers",type=int,default=1,help="Number of workers used for loading data",)
  parser.add_argument("-allones_concept",action="store_true",help="Change concept dataset to ones corresponding to class",)
  # arguments for CBL
  parser.add_argument("-crop_to_concept_prob",type=float,default=0.0,help="Probability of cropping to concept granuality",)
  parser.add_argument("-cbl_confidence_threshold",type=float,default=0.15,help="Confidence threshold for bouding boxes to use",)
  parser.add_argument("-cbl_hidden_layers",type=int,default=1,help="how many hidden layers to use in the projection layer",)
  parser.add_argument("-cbl_batch_size",type=int,default=32,help="Batch size used when fitting projection layer",)
  parser.add_argument("-cbl_epochs",type=int,default=20,help="how many steps to train the projection layer for",)
  parser.add_argument("-cbl_weight_decay",type=float,default=1e-5,help="weight decay for training the projection layer",)
  parser.add_argument("-cbl_lr",type=float,default=5e-4,help="learning rate for training the projection layer",)
  parser.add_argument("-cbl_loss_type", choices=["bce", "twoway"],default="bce",help="Loss type for training CBL",)
  parser.add_argument("-cbl_twoway_tp",type=float, default=4.0,help="TPE hyperparameter for TwoWay CBL loss",)
  parser.add_argument("-cbl_pos_weight",type=float,default=1.0,help="loss weight for positive examples",)
  parser.add_argument("-cbl_auto_weight",action="store_true",help="whether to automatically weight positive examples",)
  parser.add_argument("-cbl_finetune",action="store_true",help="Enable finetuning backbone in CBL training",)
  parser.add_argument("-cbl_bb_lr_rate",type=float,default=1,help="Rescale the learning rate of backbone during finetuning",)
  parser.add_argument("-cbl_optimizer",choices=["adam", "sgd"],default="sgd",help="Optimizer used in CBL training.",)
  parser.add_argument("-cbl_scheduler",choices=[None, "cosine"],default=None,help="Scheduler used in CBL training.",)
  # arguments for SAGA
  parser.add_argument("-saga_batch_size",type=int,default=512,help="Batch size used when fitting final layer",)
  parser.add_argument("-saga_step_size", type=float, default=0.1, help="Step size for SAGA")
  parser.add_argument("-saga_lam",type=float,default=0.0007,help="Sparsity regularization parameter, higher->more sparse",)
  parser.add_argument("-saga_n_iters",type=int,default=2000,help="How many iterations to run the final layer solver for",)
  parser.add_argument("-seed", type=int, default=42, help="Random seed for reproducibility")
  parser.add_argument("-dense", action="store_true", help="train with dense or sparse method")
  parser.add_argument("-dense_lr",type=float,default=0.001,help="Learning rate for the dense final layer training",)
  parser.add_argument("-data_parallel", action="store_true")
  parser.add_argument("-visualize_concepts", action="store_true", help="Visualize concepts")
  
  return parser

def parse_resnetcbm_args(parser, args):
  model = args.model
  ds = args.dataset.split("_")[0]
  parser.add_argument("-device", type=str, default="cuda", help="Which device to use")
  parser.add_argument("-batch_size", type=int, default=512, help="Batch size used when saving model/CLIP activations")
  parser.add_argument("-saga_batch_size", type=int, default=256, help="Batch size used when fitting final layer")
  parser.add_argument("-backbone", type=str, default="resnet18", help="Which ResNet pretrained model to use as backbone", choices=['resnet18', 'resnet34'])
  parser.add_argument("-unfreeze", type=int, default=1, help="Number of conv layers to unfreeze from the pretrained model")
  parser.add_argument("-num_c", type=int, default=64, help="Number of concepts to learn when unsupervised")
  # Training
  parser.add_argument("-optimizer", type=str, default="adam", help="Which optimizer to use", choices=['adam', 'adamw', 'sgd'])
  parser.add_argument("-lr", type=float, default=0.001, help="Learning rate")
  parser.add_argument('-n_epochs','-epochs','-e', type=int, required=True, help="Number of epochs to train the model.")
  parser.add_argument("-scheduler_type", type=str, default="plateau", help="Which scheduler to use", choices=['plateau', 'step'])
  parser.add_argument("-scheduler_kwargs", type=dict, default={}, help="Scheduler kwargs")
  parser.add_argument("-optimizer_kwargs", type=dict, default={}, help="Optimizer kwargs")
  parser.add_argument("-balancing_weight", type=float, default=0.4, help="Weight for balancing the loss")
  parser.add_argument("-patience", type=int, default=10, help="Patience for early stopping")
  parser.add_argument("-dropout_prob", type=float, default=0.01, help="Dropout probability")
  parser.add_argument("-val_interval", type=int, default=1, help="Validation interval, every n epochs do validation")
  parse_glm_args(parser, args)
  return parser

def parse_llamaoracle_args(parser,args):
  parser.add_argument("-start_idx", type=int, default=None, help="Which index of the dataset to start from when quering the oracle")
  parser.add_argument("-end_idx", type=int, default=None, help="Which index of the dataset to start from when quering the oracle")
  # Add the ResNet CBM arguments since it uses that as a backbone
  model = args.model
  ds = args.dataset.split("_")[0]
  parser.add_argument("-device", type=str, default="cuda", help="Which device to use")
  parser.add_argument("-batch_size", type=int, default=512, help="Batch size used when saving model/CLIP activations")
  parser.add_argument("-saga_batch_size", type=int, default=256, help="Batch size used when fitting final layer")
  parser.add_argument("-backbone", type=str, default="resnet18", help="Which ResNet pretrained model to use as backbone", choices=['resnet18', 'resnet34'])
  parser.add_argument("-unfreeze", type=int, default=1, help="Number of conv layers to unfreeze from the pretrained model")
  parser.add_argument("-num_c", type=int, default=64, help="Number of concepts to learn when unsupervised")
  # Training
  parser.add_argument("-optimizer", type=str, default="adam", help="Which optimizer to use", choices=['adam', 'adamw', 'sgd'])
  parser.add_argument("-lr", type=float, default=0.001, help="Learning rate")
  parser.add_argument("-n_epochs","-e", type=int, default=1000, help="Number of epochs to train for")
  parser.add_argument("-scheduler_type", type=str, default="plateau", help="Which scheduler to use", choices=['plateau', 'step'])
  parser.add_argument("-scheduler_kwargs", type=dict, default={}, help="Scheduler kwargs")
  parser.add_argument("-optimizer_kwargs", type=dict, default={}, help="Optimizer kwargs")
  parser.add_argument("-balancing_weight", type=float, default=0.4, help="Weight for balancing the loss")
  parser.add_argument("-patience", type=int, default=10, help="Patience for early stopping")
  parser.add_argument("-dropout_prob", type=float, default=0.01, help="Dropout probability")
  parser.add_argument("-val_interval", type=int, default=1, help="Validation interval, every n epochs do validation")
  parse_glm_args(parser, args)
  return parser