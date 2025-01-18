from config import CONCEPT_SETS, SAVED_MODELS_FOLDER, ACTIVATIONS_PATH

'''
DEFINE HERE THE PARSE FN FOR EACH TRAINING MODEL

The name needs to be as follows: parse_<model_name>_args(parser, args)
e.g. "def parse_lfcbm_args(parser, args)"

'''

def parse_lfcbm_args(parser, args):
  model = args.model
  ds = args.dataset.split("_")[0]
  parser.add_argument("--concept_set", type=str, default=f"{CONCEPT_SETS[ds]}/handmade.txt", 
                      help="Path to the concept set name file to use")
  parser.add_argument("--backbone", type=str, default="clip_RN50", help="Which pretrained model to use as backbone")
  parser.add_argument("--clip_name", type=str, default="ViT-B/16", help="Which CLIP model to use")

  parser.add_argument("--device", type=str, default="cuda", help="Which device to use")
  parser.add_argument("--batch_size", type=int, default=258, help="Batch size used when saving model/CLIP activations")
  parser.add_argument("--saga_batch_size", type=int, default=256, help="Batch size used when fitting final layer")
  parser.add_argument("--proj_batch_size", type=int, default=50000, help="Batch size to use when learning projection layer")

  parser.add_argument("--feature_layer", type=str, default='layer4', 
                      help="Which layer to collect activations from. Should be the name of second to last layer in the model")
  parser.add_argument("--activation_dir", type=str, default='shared', help="save location for backbone and CLIP activations")
  parser.add_argument("--save_dir", type=str, default=SAVED_MODELS_FOLDER[model], help="where to save trained models")
  parser.add_argument("--clip_cutoff", type=float, default=0.25, help="concepts with smaller top5 clip activation will be deleted")
  parser.add_argument("--proj_steps", type=int, default=1000, help="how many steps to train the projection layer for")
  parser.add_argument("--interpretability_cutoff", type=float, default=0.45, help="concepts with smaller similarity to target concept will be deleted")
  parser.add_argument("--lam", type=float, default=0.0007, help="Sparsity regularization parameter, higher->more sparse")
  parser.add_argument("--n_iters", type=int, default=1000, help="How many iterations to run the final layer solver for")
  parser.add_argument("--print", action='store_true', help="Print all concepts being deleted in this stage")
  
  return parser

def parse_vlgcbm_args(parser, args):
  raise NotImplementedError("Not implemented yet")

def parse_resnetcbm_args(parser, args):
  raise NotImplementedError("Not implemented yet")

def parse_labo_args(parser, args):
  raise NotImplementedError("Not implemented yet")