from models import get_model
from config import SAVED_MODELS_FOLDER
from core.concept_quality import initialize_CQA
import argparse
import sys
from loguru import logger
import datetime

#TODO: Fix DCI 235hudhf train and test error
#TODO: Set -model argparser so that it can be used to load the model
#TODO: Implement vlgcbm_utils asdgteyt34tdfs for plotting annotations
def main():
  parser = argparse.ArgumentParser(description="Test for lf-cbm")
  parser.add_argument("-force", action="store_true", help="Force the computation from scratch")
  parser.add_argument("-logger", type=str, default="DEBUG", help="Logging level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
  parser.add_argument("-model", type=str, help="Model to use", choices=["vlgcbm", "lfcbm","resnetcbm"])
  args = parser.parse_args()
  ''' Set up logger'''
  logger.remove()
  def my_filter(record):
    return record["level"].no >= logger.level(args.logger).no
  logger.add(sys.stderr, filter=my_filter)
  ''' ------------------------------------- '''
  logger.debug(f"Arguments: {vars(args)}")
  base = SAVED_MODELS_FOLDER['lfcbm']
  #fold="celeba_cbm_2024_08_31_10_29"
  fold = "resnetcbm"
  #fold = "lfcbm_celeba_2025_01_13_15_48"
  CQA = initialize_CQA(base + fold, args, split = 'test', force_from_scratch = args.force)
  if CQA.dci is None:
    logger.info("Computing DCI...")
    CQA.DCI(0.8)
  print(CQA.dci)
  #print(CQA.metrics())
  #CQA.get_classification_report()
  #print(CQA.classification_report)
  
  #print(CQA.output)

  #CQA.save_im_as_img(base + fold, "importance_matrix", "Importance Matrix")

if __name__ == "__main__":
    start = datetime.datetime.now()
    main()
    end = datetime.datetime.now()
    print('\n ### Total time taken: ', end - start)
    print('\n ### Closing ###')
