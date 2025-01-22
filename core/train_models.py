from loguru import logger
import os
from models.training import get_trainer
from config import SAVED_MODELS_FOLDER, folder_naming_convention
from utils.args_utils import save_args


def run(args):
    # Save folder
    folder_name = folder_naming_convention(args)
    if args.save_dir is None:
        args.save_dir = os.path.join(SAVED_MODELS_FOLDER[args.model],folder_name)
    logger.debug(f"Created folder: {args.save_dir}")
    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(f"Starting training model {args.model}")
    train_fn = get_trainer(args)
    final_args = train_fn(args)
    save_args(final_args)
    return