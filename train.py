import argparse
from loguru import logger 
import importlib
import sys
import datetime
import setproctitle, socket, uuid
from core.train_models import run
import json

# TODO: add seed for reproducibility
# TODO: change llamaoracle name to oracle
'''
Example:
python train.py -model vlgcbm -dataset celeba -e 5

'''

def parse_args():
    parser = argparse.ArgumentParser(description="Dynamic flags based on initial flag value.")
    
    # Add the primary flag
    parser.add_argument('-model','-m', required=True, type=str, choices=['lfcbm', 'resnetcbm','llamaoracle','vlgcbm','labo'], help="Specify the model to train.")
    parser.add_argument('-logger', type=str, default="DEBUG", help="Logging level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("-dataset",'-d', type=str, default="celeba", help="Dataset to use")
    parser.add_argument("-config", type=str, default=None, help="Path to a config file for setting all the parameters in a json file")
    parser.add_argument("-save_dir", type=str, default=None, help="Folder where to save the model")
    parser.add_argument("-wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("-seed", type=int, default=42, help="Set the random seed")
    parser.add_argument("-resume", type=str, default=None, help="Path to a model to resume training")
    # Parse known arguments to determine the value of --model
    args, remaining_args = parser.parse_known_args()
    
    ''' Set up logger'''
    logger.remove()
    def my_filter(record):
        return record["level"].no >= logger.level(args.logger).no
    logger.add(sys.stderr, filter=my_filter)
    ''' ------------------------------------- '''

    args_dict = vars(args)
    # Define subparsers for dynamic flags
    # Add model-specific args
    argparser_module = importlib.import_module(f"utils.argparser")
    try:
        parse_model_args = getattr(argparser_module, f"parse_{args.model}_args")
    except AttributeError:
        raise ValueError(f"Argparser method for {args.model} not defined. Do it in CQA/utils/argparser.py")
    
    model_parser = argparse.ArgumentParser(description="Model specific flags")
    model_parser = parse_model_args(model_parser,args)
    if args.config is not None:
      with open(args.config, "r") as f:
          config_arg = json.load(f)
      model_parser.set_defaults(**config_arg)
    sub_args = model_parser.parse_args(remaining_args)
   
    # Combine the primary args and the sub_args
    args_dict.update(vars(sub_args))
    args = argparse.Namespace(**args_dict)

    args.time = datetime.datetime.now().strftime("%H_%M")
    args.date = datetime.datetime.now().strftime("%Y_%m_%d")
    args.conf_host = socket.gethostname()
    args.conf_jobnum = str(uuid.uuid4())
    # set job name
    setproctitle.setproctitle('{}_{}_{}'.format( args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))
    return args

if __name__ == "__main__":
    start = datetime.datetime.now()
    args = parse_args()
    run(args)
    end = datetime.datetime.now()
    print('\n ### Total time taken: ', end - start)
    print('\n ### Closing ###')
