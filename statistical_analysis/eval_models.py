import os
import pickle
from loguru import logger
from CQA.core.concept_quality import CONCEPT_QUALITY
from CQA.utils.args_utils import load_args_from_path
from CQA.core.concept_quality import initialize_CQA_v2
from CQA.config import CONCEPT_SETS
from CQA.config import ACTIVATIONS_PATH
import sys

logger.remove()  # Remove the default handler
logger.add(sys.stderr, level="INFO")  # Add handler with INFO level

FOLDER = '/mnt/cimec-storage6/users/nicola.debole/home/CQA/rebuttal_models'
DATASETS = ['celeba']
MODELS = ['labo','lfcbm', 'oracle', 'resnetcbm', 'vlgcbm']

for d in DATASETS:
    for m in MODELS:
        path = os.path.join(FOLDER,d,m)
        runs = os.listdir(path)
        for r in runs:
            m_path = os.path.join(path, r)
            args = load_args_from_path(os.path.join(m_path))
            args.force = True
            args.activation_dir = '/mnt/cimec-storage6/users/nicola.debole/home/CQA/data/activations'
            ACTIVATIONS_PATH['shared'] = args.activation_dir
            args.concept_set = CONCEPT_SETS[d]
            args.eval_seed = 10
            CQA = initialize_CQA_v2(args.load_dir, args, split = 'test')
            CQA.get_classification_report()
            CQA.concept_metrics()
            CQA.save()
            CQA.compute_leakage()
            CQA.save()
            CQA.compute_ois()
            CQA.save()
            CQA.DCI(0.8)
            CQA.save()
            CQA.dump_metrics()
            logger.info(f"{d}-{m}")
            