from loguru import logger
import os, traceback
from config import SAVED_MODELS_FOLDER
from models import get_model
from config import SAVED_MODELS_FOLDER
from core.concept_quality import initialize_CQA


def eval_model(args):
    try:
        CQA = initialize_CQA(args.load_dir, args, split = 'test', force_from_scratch = args.force)
        logger.debug(f"Args: {CQA.main_args}")
        if CQA.main_args.wandb:
            import wandb
            wandb.init(project="Concept Quality Analysis", id=args.run_idp, resume="allow")
            # 28-13-25
            # Define the metric to allow manual step logging
            wandb.define_metric("concept_accuracy", step_metric="manual_step")

        if CQA.main_args.dci or CQA.main_args.all:
            logger.info("Computing DCI...")
            if CQA.dci is not None and not CQA.main_args.force:
                print(CQA.dci)
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
            CQA.metrics()
            CQA.save()

        if CQA.main_args.label_metrics or CQA.main_args.all:
            CQA.get_classification_report()
            print(CQA.classification_report)
            CQA.save()
            #CQA.metrics()
            #print(CQA.metrics())
            #print(CQA.output)
        
        if CQA.main_args.wandb:
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