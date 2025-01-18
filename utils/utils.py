from loguru import logger
from tqdm import tqdm
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=target_mean, std=target_std),
        ]
    )
    return preprocess

def get_concept_names(file_path:str)->list[str]:
    with open(file_path) as f:
        concepts = f.read().split("\n")
    return concepts

def plot_examples(list_of_images, labels, text = '', img_per_class = 10):
    ''' Plot examples of images from each class in the dataset
    Args:
        list_of_images: list of images
        labels: list of labels corresponding to the images
        text: Name of the dataset the images are coming from
        img_per_class: number of images to be displayed for each class'''
    # Classify the images based on the task labels
    classes = {}
    for index,label in enumerate(labels):
        if label in classes:
            classes[label].append(index)
        else:
            classes[label] = [index]

    num_classes = len(classes)

    # Randomly shuffle the images in each class
    for key in classes:
        classes[key] = np.random.permutation(classes[key])
    
    # Plot the images
    fig = plt.figure(figsize=(10,10))
    plt.suptitle(f'Dataset: {text}')
    for i in range(num_classes):
        for j in range(img_per_class):
            ax = fig.add_subplot(num_classes, img_per_class, i*img_per_class+j+1)
            plt.title(f'Class: {i}')
            ax.imshow(list_of_images[classes[i][j]])
            ax.axis('off')
    plt.show()

def one_hot_concepts(concepts):
    I = np.unique(concepts)
    one_hots = []
    diag_matrix = np.eye(len(I))
    for sample in range(len(concepts)):
        for i in range(len(I)):
            if concepts[sample] == I[i]:
                one_hots.append(diag_matrix[i])
    one_hots = np.stack(one_hots)
    return one_hots