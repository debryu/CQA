import json
from loguru import logger
import os

def load_args(args):
    '''
    Load the args from the args.load_dir folder.
    Input:
        args: the args object from the main script
    Output:
        args: the updated args object
    '''
    logger.debug("Loading from {}".format(args.load_dir))
    # args only contains the load_dir
    load_dir = args.load_dir
    with open(os.path.join(args.load_dir, "args.txt")) as f:
        loaded_args = json.load(f)
    # Add the args from the model checkpoint
    for key, value in loaded_args.items():
        setattr(args, key, value)
    setattr(args, "load_dir", load_dir)
    logger.debug(args)
    return args