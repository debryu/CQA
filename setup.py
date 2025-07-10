from setuptools import setup, find_packages


        
setup(
    name='CQA',
    version='0.1',
    description='Concept Quality Analysis',
    author='Nicola Debole',
    author_email='nicola.debole@unitn.it',
    packages=find_packages(),
    install_requires=[
        "loguru",
        "medmnist",
        "h5py",
        "matplotlib",
        "wandb",
        "setproctitle",
        "pytorchcv",
        "ftfy",
        "regex",
        "ollama",
        "seaborn",
        ],  
)
