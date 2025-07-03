# Examples
These are just examples on how to train the various models. All other parameters are optional and customizable.
If you encounter errors, be sure ```config.py``` is properly set with your configuration.

## Train CBM
#### Celeba and Shapes3d
```
python train.py -model resnetcbm -dataset <DATASET> -epochs <# Epochs> -unfreeze 5
```
#### Cub
```
python train.py -model resnetcbm -backbone resnet18_cub -dataset cub -epochs <# Epochs> -unfreeze 5
```


## Train LaBo
```
python train.py -model labo -dataset <DATASET>
```

## Train LF-CBM
#### Celeba and Shapes3d
```
python train.py -model lfcbm -dataset <DATASET>
```
#### Cub
```
python train.py -model lfcbm -backbone resnet18_cub -feature_layer features.final_pool -dataset cub
```

## Train VLG-CBM
#### Celeba and Shapes3d
```
python train.py -model vlgcbm -dataset <DATASET>
```
#### Cub
```
python train.py -model vlgcbm -backbone resnet18_cub -feature_layer features.final_pool -dataset cub
```