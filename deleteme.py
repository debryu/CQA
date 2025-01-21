from datasets import get_dataset
from torchvision.datasets import CIFAR10
from config import DATASETS_FOLDER_PATHS
import datetime
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic flags based on initial flag value.")
    #parser.add_argument('-model', required=True, type=str, choices=['lfcbm', 'debug'], help="Specify the model to train.")
    #parser.add_argument('-logger', type=str, default="DEBUG", help="Logging level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("-dataset", type=str, default="celeba", help="Dataset to use")
    args = parser.parse_args()
    start = datetime.datetime.now()
    data = get_dataset('cifar10')
    print(data[0])
    end = datetime.datetime.now()
    print('\n ### Total time taken: ', end - start)
    print('\n ### Closing ###')