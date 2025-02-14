from loguru import logger
import os, traceback
from config import SAVED_MODELS_FOLDER
from models import get_model
from config import SAVED_MODELS_FOLDER, CONCEPTS_NOT_AVAILABLE
from core.concept_quality import initialize_CQA
from utils.args_utils import load_args
import shutil
import copy
# TODO: remove temporary fixes  asdg5etr and 42ohdfsa
def eval_model(arguments):
    try:
        logger.info("Eval model", arguments)
        CQA = initialize_CQA(arguments.load_dir, arguments, split = 'test')
        #logger.debug(f"main.py args: {CQA.main_args}")
        if CQA.main_args.wandb:
            import wandb
            # If it throws an error, it is because the args are not correctly loaded
            # Just uncomment and run again once, it will work later (then comment again)
            # asdg5etr
            #args = load_args(args)
            #CQA.args = args
            #CQA.save()

            # First, check if the run_id is already saved in CQA
            if hasattr(CQA,'run_id'):
                if CQA.run_id is not None:
                    arguments.run_id = CQA.run_id
            # Then try resuming a previous wandb run
            try:
                logger.info(f"Initializing wandb with run id: {arguments.run_id}")
                wandb.init(project="Concept Quality Analysis", id=arguments.run_id, resume="allow")
                wandb.define_metric("single_step")
                wandb.define_metric("concept_accuracy", step_metric="manual_step")
                wandb.define_metric("label_accuracy", step_metric="single_step")
                wandb.define_metric("label_f1", step_metric="single_step")
                wandb.define_metric("disentanglement", step_metric="single_step")
                wandb.define_metric("avg_concept_accuracy", step_metric="single_step")
                wandb.define_metric("avg_concept_f1", step_metric="single_step")
                pass
            # If this is not possible, then create a new one.
            except Exception as e:
                if 'timed out' in str(e) or "has no attribute 'run_id'" in str(e):
                    logger.error("Cannot load existing run. Creating a new run.")
                    if str(arguments.load_dir).endswith("/") or str(arguments.load_dir).endswith("\\"):
                        logger.debug(f"Creating new run '{arguments.load_dir}'")
                        wandb_run = wandb.init(project="Concept Quality Analysis", name=arguments.load_dir,config=arguments)
                    else:
                        logger.debug(f"Creating new run '{os.path.basename(arguments.load_dir)}'")
                        wandb_run = wandb.init(project="Concept Quality Analysis", name=os.path.basename(arguments.load_dir),config=arguments)
                    
                    CQA.run_id = wandb_run.id    
                    wandb.define_metric("single_step")
                    wandb.define_metric("concept_accuracy", step_metric="manual_step")
                    wandb.define_metric("label_accuracy", step_metric="single_step")
                    wandb.define_metric("label_f1", step_metric="single_step")
                    wandb.define_metric("disentanglement", step_metric="single_step")
                    wandb.define_metric("avg_concept_accuracy", step_metric="single_step")
                    wandb.define_metric("avg_concept_f1", step_metric="single_step")

                else:
                    logger.error("Error in initializing wandb")
                    logger.error(traceback.format_exc())
                    logger.warning("Disabling wandb")
                    CQA.main_args.wandb = False
            # 28-13-25
            # Define the metric to allow manual step logging
            
        # Set the criteria for every metric
        criteria_dci = CQA.args.dataset not in CONCEPTS_NOT_AVAILABLE
        criteria_concept_metrics = CQA.args.dataset not in CONCEPTS_NOT_AVAILABLE
        criteria_label_metrics = True
        print(CQA.metrics)
        if (CQA.main_args.dci or CQA.main_args.all) and criteria_dci:
            logger.info("Computing DCI...")
            if CQA.dci is not None:
                CQA.save_im_as_img(CQA.main_args.load_dir, "importance_matrix.png", "DCI Importance Matrix")
                print(CQA.dci)
                # 42ohdfsa
                CQA.metrics['disentanglement'] = CQA.dci['disentanglement']
                CQA.metrics['completeness'] = CQA.dci['completeness']
                #
            else:
                try:
                    CQA.DCI(0.8)
                    print(CQA.dci['disentanglement'])
                    CQA.save_im_as_img(CQA.main_args.load_dir,  "importance_matrix.png", "DCI Importance Matrix")
                    CQA.save()
                except:
                    logger.error("Error in computing DCI")
                    logger.error(traceback.format_exc())
            
        if (CQA.main_args.concept_metrics or CQA.main_args.all) and criteria_concept_metrics:
            if 'avg_concept_accuracy' not in CQA.metrics:
                CQA.concept_metrics()
            #CQA.save()

        if (CQA.main_args.label_metrics or CQA.main_args.all) and criteria_label_metrics:
            if 'label_accuracy' not in CQA.metrics:
                CQA.get_classification_report()
            print(CQA.classification_report)
            #CQA.save()
            #CQA.metrics()
            #print(CQA.metrics())
            #print(CQA.output)
        
        if CQA.main_args.wandb:
            CQA.log_metrics()
            wandb.finish()
        CQA.save()

    except Exception as e:
        # Try only loading the args. If it is not possible, delete the foder!
        try:
            load_args(arguments)
        except:
            logger.warning(f"!!!    USER ACTION REQUESTED   !!!")
            confirmation = input(f"There is no 'args.txt' file. Do you want to delete '{arguments.load_dir}' and all its contents? (yes/no): ").strip().lower()
            print(confirmation)
            if str(confirmation) == "yes" or str(confirmation) == "y":
                shutil.rmtree(arguments.load_dir)
                logger.info(f"Deleted {arguments.load_dir}")
                return
            else:
                logger.info("Deletion canceled.")
                CQA.save()
        
        logger.error(f"Error in initializing CQA {arguments.load_dir}:\n{e}")
        logger.error(traceback.format_exc())
        logger.warning(f"!!!    USER ACTION REQUESTED   !!!")
        confirmation = input(f"Do you want to delete '{arguments.load_dir}' and all its contents? (yes/no): ").strip().lower()
        print(confirmation)
        if str(confirmation) == "yes" or str(confirmation) == "y":
            shutil.rmtree(arguments.load_dir)
            logger.info(f"Deleted {arguments.load_dir}")
        else:
            logger.info("Deletion canceled.")
            CQA.save()
    return 
    

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
            temp_main_args = copy.deepcopy(args)
            temp_main_args.load_dir = os.path.join(path, model)
            eval_model(temp_main_args)
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