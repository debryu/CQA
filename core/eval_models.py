from loguru import logger
import os, traceback
from config import SAVED_MODELS_FOLDER
from models import get_model
from config import SAVED_MODELS_FOLDER
from core.concept_quality import initialize_CQA
from utils.args_utils import load_args

# TODO: remove temporary fixes  asdg5etr and 42ohdfsa
def eval_model(args):
    try:
        CQA = initialize_CQA(args.load_dir, args, split = 'test', force_from_scratch = args.force)
        logger.debug(f"Args: {CQA.main_args}")
        if CQA.main_args.wandb:
            import wandb
            # If it throws an error, it is because the args are not correctly loaded
            # Just uncomment and run again once, it will work later (then comment again)
            # asdg5etr
            args = load_args(args)
            CQA.args = args
            CQA.save()
            #
            logger.info(f"Initializing wandb with run id: {args.run_id}")
            try:
                wandb.init(project="Concept Quality Analysis", id=args.run_id, resume="allow")
                wandb.define_metric("concept_accuracy", step_metric="manual_step")
            except:
                logger.error("Error in initializing wandb")
                logger.warning("Disabling wandb")
                CQA.main_args.wandb = False
            # 28-13-25
            # Define the metric to allow manual step logging
            

        if CQA.main_args.dci or CQA.main_args.all:
            logger.info("Computing DCI...")
            if CQA.dci is not None:
                print(CQA.dci)
                # 42ohdfsa
                CQA.metrics['disentanglement'] = CQA.dci['disentanglement']
                CQA.metrics['completeness'] = CQA.dci['completeness']
                #
            else:
                try:
                    CQA.DCI(0.8)
                    print(CQA.dci['disentanglement'])
                    CQA.save_im_as_img(args.load_dir, "importance_matrix", "Importance Matrix")
                    CQA.save()
                except:
                    logger.error("Error in computing DCI")
                    logger.error(traceback.format_exc())
            
        if CQA.main_args.concept_metrics or CQA.main_args.all:
            CQA.concept_metrics()
            CQA.save()

        if CQA.main_args.label_metrics or CQA.main_args.all:
            CQA.get_classification_report()
            print(CQA.classification_report)
            CQA.save()
            #CQA.metrics()
            #print(CQA.metrics())
            #print(CQA.output)
        
        if CQA.main_args.wandb:
            CQA.log_metrics()
            wandb.finish()

    except Exception as e:
        logger.error(f"Error in initializing CQA {args.load_dir}:\n{e}")
        logger.error(traceback.format_exc())
        
    

def eval_all_models(args):
    completed = []
    for model, path in SAVED_MODELS_FOLDER.items():
        logger.debug(f"Model: {model}, Path: {path}")
        if path in completed:
            logger.debug(f"Skipping {path} as it is already completed")
            continue
        models = os.listdir(path)
        for model in models:
            logger.info(f"Loading model: {path}/{model}")
            args.load_dir = os.path.join(path, model)
            eval_model(args)
        completed.append(path)
    return

def CQ_Analysis(args):
    if args.load_dir is not None:
        logger.info(f"Loading model from {args.load_dir}")
        eval_model(args)
    else:
        logger.info(f"Loading all the models from config")
        eval_all_models(args)
    return