from loguru import logger
from tqdm import tqdm
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import wandb 
import random
import os, requests
from urllib.parse import urlparse
from typing import List

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def get_concept_names(file_path:str)->List[str]:
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

def log_train(epoch, args, train_loss = None, val_loss = None, dict = None):
    logs = {"epoch":epoch}
    if args.wandb:
        if train_loss is not None:
            logs["train_loss"] = train_loss
        if val_loss is not None:
            logs["val_loss"] = val_loss
        if dict is not None:
            logs.update(dict)
        wandb.log(logs)
    logger.info(f"Train Loss: {train_loss}, Val Loss: {val_loss}")


# Function to download images
def download_image(url, file_path):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    }
    try:
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()  # Raise an error for bad responses
        
        # Extract filename from URL
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        
        if not filename:  # If URL does not have a filename, generate one
            filename = f"image_{hash(url)}.jpg"
        
        # Save image
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        
        logger.debug(f"Downloaded: {filename}")
        return True
    except Exception as e:
        logger.warning(f"Failed to download {url}: {e}")
        return False


