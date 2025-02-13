from loguru import logger
import os
from models.training import get_trainer, get_last_layer_trainer
from config import SAVED_MODELS_FOLDER, folder_naming_convention
from utils.args_utils import save_args
from utils.utils import set_seed
from utils.args_utils import load_args
import copy
import wandb

def run(args):
    if args.resume is not None:
        t_args = copy.deepcopy(args)
        t_args.load_dir = args.resume
        M_args = load_args(t_args)
        print(M_args)
        train_fn = get_last_layer_trainer(M_args) 
        final_args = train_fn(M_args)
        save_args(final_args)
        return
    # Save folder
    folder_name = folder_naming_convention(args)
    if args.save_dir is None:
        args.save_dir = os.path.join(SAVED_MODELS_FOLDER[args.model],folder_name)
    logger.debug(f"Created folder: {args.save_dir}")
    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(f"Starting training model {args.model}")
    if args.seed:
        logger.info(f"#- Run {args.seed} -#")
        set_seed(args.seed)
    if args.wandb:
        wand_run = wandb.init(
            # Set the project where this run will be logged
            project=f"Concept Quality Analysis",
            name=folder_name,
            # Track hyperparameters and run metadata
            config=args
        )
        args.run_id = wand_run.id

    train_fn = get_trainer(args)
    final_args = train_fn(args)
    if args.wandb:
        wandb.config.update(final_args, allow_val_change=True)
        wandb.finish()
    save_args(final_args)
    return