import argparse
import sys
from loguru import logger
import datetime

from core.eval_models import CQ_Analysis
#TODO: Fix DCI 235hudhf train and test error
#TODO: Set -model argparser so that it can be used to load the model
#TODO: Implement vlgcbm_utils asdgteyt34tdfs for plotting annotations
#TODO: Implement flags for computing DCI, concept wise metrics, label metrics
def main():
  parser = argparse.ArgumentParser(description="Evaluate models")
  parser.add_argument("-folder", type=str, default=None, help="Directory containing multiple models to test")
  parser.add_argument("-load_dir", type=str, default=None, help="Load directory for the model. If not provided, the path in the config will be used")
  parser.add_argument("-force", action="store_true", help="Force the computation from scratch")
  parser.add_argument("-all", action="store_true", help="Compute all possible metrics")
  parser.add_argument("-wandb", action="store_true", help="Logs on wandb")
  parser.add_argument("-dci", action="store_true", help="Compute DCI")
  parser.add_argument("-concept_metrics", action="store_true", help="Compute metrics concept wise")
  parser.add_argument("-label_metrics", action="store_true", help="Compute classification report")
  parser.add_argument("-logger", type=str, default="DEBUG", help="Logging level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
  #parser.add_argument("-model", type=str, help="Model to use", choices=["vlgcbm", "lfcbm","resnetcbm"])
  args = parser.parse_args()
  ''' Set up logger'''
  logger.remove()
  def my_filter(record):
    return record["level"].no >= logger.level(args.logger).no
  logger.add(sys.stderr, filter=my_filter)
  ''' ------------------------------------- '''
  CQ_Analysis(args)
  return

if __name__ == "__main__":
    start = datetime.datetime.now()
    main()
    end = datetime.datetime.now()
    print('\n ### Total time taken: ', end - start)
    print('\n ### Closing ###')
