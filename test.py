from models import get_model
from core.concept_quality import initialize_CQA
import argparse
import sys
from loguru import logger



def main():
  parser = argparse.ArgumentParser(description="Test for lf-cbm")
  parser.add_argument("--force", action="store_true", help="Force the computation from scratch")
  parser.add_argument("--level", type=str, default="INFO", help="Logging level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
  args = parser.parse_args()
  ''' Set up logger'''
  logger.remove()
  def my_filter(record):
    return record["level"].no >= logger.level(args.level).no
  logger.add(sys.stderr, filter=my_filter)
  ''' ------------------------------------- '''

  base = "C:\\Users\\debryu\\Desktop\\VS_CODE\\HOME\\ML\\Tirocinio\\interpreter\\data\\don use\\old_lfcbm/"
  #fold="celeba_cbm_2024_08_31_10_29"
  fold = "celeba_cbm_2024_09_12_00_00"
  CQA = initialize_CQA(base + fold, args, split = 'test', force_from_scratch = args.force)
  CQA.DCI(0.8)
  print(CQA.dci)
  #CQA.save_im_as_img(base + fold, "importance_matrix", "Importance Matrix")

if __name__ == "__main__":
  main()
