import json
from loguru import logger
import os
import argparse

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

def load_args_from_path(path):
    '''
    Load the args from the args.load_dir folder.
    Input:
        args: the args object from the main script
    Output:
        args: the updated args object
    '''
    args = argparse.Namespace()
    logger.debug("Loading from {}".format(path))
    # args only contains the load_dir
    with open(os.path.join(path, 'args.txt')) as f:
        loaded_args = json.load(f)
    # Add the args from the model checkpoint
    for key, value in loaded_args.items():
        setattr(args, key, value)
    setattr(args, "load_dir", path)
    logger.debug(args)
    return args

def save_args(args: argparse.Namespace):
    logger.debug("Saving args to {}".format(args.save_dir))
    with open(os.path.join(args.save_dir, "args.txt"), 'w') as f:
        json.dump(vars(args), f)
    return