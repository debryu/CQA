from torch.utils.data.sampler import Sampler
from tqdm import tqdm
import h5py
import numpy as np
import torch
import os
from CQA.utils.utils import plot_examples, one_hot_concepts
'''
May be useful
def compute_concept_frequencies(dataset:str, split:str = 'train'):
    
    #dataset: str, the name of the dataset to use
    #split: str, the split to use
    
    logger.debug(f"Computing concept and label frequencies for {dataset}...")
    data = get_dataset(dataset, split = split)
    concepts = []
    labels = []
    for i in tqdm(range(len(data))):
        _, c, l = data[i]
        concepts.append(c)
        labels.append(l)
    concepts = torch.stack(concepts)
    labels = torch.stack(labels)
    c_freq = concepts.sum(dim = 0) / len(concepts)
    c_freq = c_freq.cpu().numpy().tolist()
    l_freq = labels.sum(dim = 0) / len(labels)
    l_freq = l_freq.cpu().numpy().tolist()
    return c_freq, l_freq
'''
def compute_imbalance(dataset):
  n = len(dataset)
  imbalance_ratio = []
  n_attr = dataset[0][1].shape[0]
  print(f"Number of attributes:{n_attr}")
  attr_idx = -1
  multiple_attr = True
  if attr_idx >= 0:
      n_attr = 1
  if multiple_attr:
      n_ones = [0] * n_attr
      total = [n] * n_attr
  else:
      n_ones = [0]
      total = [n * n_attr]
  for d in tqdm(dataset, desc='Computing imbalance'):
      _, concepts, labels =  d
      if multiple_attr:
          for i in range(n_attr):
              n_ones[i] += concepts[i]
      else:
          if attr_idx >= 0:
              n_ones[0] += concepts[attr_idx]
          else:
              n_ones[0] += sum(labels)
  for j in range(len(n_ones)):
      imbalance_ratio.append(total[j]/n_ones[j] - 1)
  if not multiple_attr: #e.g. [9.0] --> [9.0] * 312
      imbalance_ratio *= n_attr
  return imbalance_ratio

def shapes_3d_base(base_path='./data/shapes3d/3dshapes.h5'):
    print('Loading the dataset...')
    print(base_path)
    with h5py.File(base_path, 'r') as f:

        images = f['images']
        concepts = f['labels']

        images   = images[()]
        concepts = concepts[()]
        labels   = np.copy(concepts)
    
    # Description of the dataset concepts (3D shapes)
    _FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                     'orientation']
    _NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 
                          'scale': 8, 'shape': 4, 'orientation': 15}
    '''     
            0.0 = red
            0.1 = orange
            0.2 = yellow
            0.7 = blue
    '''
    #print(labels[:,2])
    ## START CONVERTING LABELS IN ONE-HOT ENCODING

    # for j in range(labels.shape[1]):
    #     column = labels[:,j]
        
    #     numbers = column
    #     I = np.unique(column)

    #     result = []
    #     for num in numbers:
    #         if num in I:
    #             result.append( np.where(I == num)[0])
    #     labels[:,j] = np.array( result ).reshape(-1)
    # labels = labels.astype(int)

    ## CREATE LABELS FOR CLASSIFYING THE PILL SHAPE, RED, any size, any orientation
    predictions = []
    for j in range(labels.shape[0]):
        if labels[j,4] == 3 and labels[j,2] == 0.0:
            predictions.append(1)
        else:
            predictions.append(0)

    labels = np.array(predictions)
    #print(labels.shape)

    # print(labels[10000:10020])

    # quit()

    # for i in range(concepts.shape[-1]):
    #     c = concepts[:,i]
    #     cmax = np.max(c)
    #     cmin = np.min(c)

    #     c = (c - cmin) / (cmax -cmin) #- 0.5

    #     concepts[:,i] = c
    encoding = 'one-hot'
    if encoding == 'binary-concepts':
        is_pill = (concepts[:,4] == 3).astype(int).reshape(-1,1)
        is_red = (concepts[:,2] == 0).astype(int).reshape(-1,1)
        is_orange = (concepts[:,2] == 0.1).astype(int).reshape(-1,1)
        is_yellow = (concepts[:,2] == 0.2).astype(int).reshape(-1,1)
        is_blue = (concepts[:,2] == 0.7).astype(int).reshape(-1,1)
        preprocess_concepts = np.hstack((is_pill, is_red, is_orange, is_yellow, is_blue))

    if encoding == 'unique-id':
        concepts = concepts[:,:5] # Remove the orientation concept
        shape = concepts.shape
        preprocess_concepts = np.zeros(shape)
        # Encode the concepts as integer numbers
        for j in range(concepts.shape[1]):
            if j in [0,3]: # Skipping 1,2 because they are already fixed by the 0 step, and 4 because it is in the correct format
                column = concepts[:,j]
                I = np.unique(column)
                for i in range(len(I)):
                    preprocess_concepts[concepts == I[i]] = i
        
        for sample in range(concepts.shape[0]):
            preprocess_concepts[sample,4] = concepts[sample,4]
    
    if encoding == 'one-hot':
        one_hots_to_concat = []
        concepts = concepts[:,:5] # Remove the orientation concept
        shape = concepts.shape
        for i in range(shape[1]):
            colunm = concepts[:,i]
            one_hots_to_concat.append(one_hot_concepts(colunm))
        preprocess_concepts = np.hstack(one_hots_to_concat)

    #print(preprocess_concepts.shape)
    #for i in range(5):
    #    print(np.unique(preprocess_concepts[:,i]), i)
    
    return images, preprocess_concepts, labels

def create_dataset(base_path = './data/shapes3d/shapes3d.h5', args=None, create_ood = False):
    print('Creating the dataset to be saved...')
    # Load the dataset
    images, concepts, labels = shapes_3d_base(os.path.join(base_path, '3dshapes.h5'))
    # Split the dataset in train, val, test, ood
    train_split = 0.7
    val_split = 0.1
    test_split = 0.2
    ood_split = 0.0

    # Extract OOD samples
    # For example define the OOD samples as the ones that contain anything yellow
    # The dimensions are now:
        # [0-10): floor_hue
        # [10-20): wall_hue
        # [20-30): object_hue
        # [30-38): scale
        # [38-42): shape
    # In particular for the first 3 dimensions, the yellow is encoded as one in the third dimension (so at index 2)
        
    in_distribution = []
    out_distribution = []
    if not create_ood:
        in_distribution = range(len(images))
    for i, image in enumerate(images):
        indexes = [2,3,4]
        # 2 = light green
        # 3 = green
        # 4 = aqua green
        if create_ood:
            ood = False
            for index in indexes:
                if concepts[i,index] == 1.0 or concepts[i,10 + index] == 1.0 or concepts[i,20 + index] == 1.0:
                    ood = True
            if ood:
                out_distribution.append(i)
            else:
                in_distribution.append(i)

    num_images = len(in_distribution)
    # Shuffle the dataset
    permutation = np.random.permutation(in_distribution)
    rnd_images = images[permutation]
    rnd_concepts = concepts[permutation]
    rnd_labels = labels[permutation]

    # Train
    train_images = rnd_images[:int(train_split*num_images)]
    train_concepts = rnd_concepts[:int(train_split*num_images)]
    train_labels = rnd_labels[:int(train_split*num_images)]

    # Val
    val_images = rnd_images[int(train_split*num_images):int((train_split+val_split)*num_images)]
    val_concepts = rnd_concepts[int(train_split*num_images):int((train_split+val_split)*num_images)]
    val_labels = rnd_labels[int(train_split*num_images):int((train_split+val_split)*num_images)]
    
    # Test
    test_images = rnd_images[int((train_split+val_split)*num_images):int((train_split+val_split+test_split)*num_images)]
    test_concepts = rnd_concepts[int((train_split+val_split)*num_images):int((train_split+val_split+test_split)*num_images)]
    test_labels = rnd_labels[int((train_split+val_split)*num_images):int((train_split+val_split+test_split)*num_images)]

    # OOD
    ood_images = images[out_distribution]
    ood_concepts = concepts[out_distribution]
    ood_labels = labels[out_distribution]

    # Visualize the dataset to make sure it is correct
    plot_examples(train_images, train_labels, text = 'train', img_per_class=10)
    plot_examples(val_images, val_labels, text = 'val', img_per_class=10)
    plot_examples(test_images, test_labels, text = 'test', img_per_class=10)
    plot_examples(ood_images, ood_labels, text = 'ood', img_per_class=10)
    answer = input("---------------[ATTENTION]-------------------\n \n \nDo you want to save the dataset?")
    if answer.lower() in ["y","yes"]:
        # Continue
        print('Saving dataset')
    else:
        # Handle "wrong" input
        print('Throwing exception')
        raise Exception('User stopped the process')

    # Save the splits
    np.save(os.path.join(base_path, 'train_split_imgs.npy'), train_images)
    np.save(os.path.join(base_path, 'train_split_cl.npy'), np.hstack((train_concepts, train_labels.reshape(-1,1)))) # Combine concepts and labels in a single array (first N columns for concepts, last one for labels)
    np.save(os.path.join(base_path, 'val_split_imgs.npy'), val_images)
    np.save(os.path.join(base_path, 'val_split_cl.npy'), np.hstack((val_concepts, val_labels.reshape(-1,1))))
    np.save(os.path.join(base_path, 'test_split_imgs.npy'), test_images)
    np.save(os.path.join(base_path, 'test_split_cl.npy'), np.hstack((test_concepts, test_labels.reshape(-1,1))))
    np.save(os.path.join(base_path, 'ood_split_imgs.npy'), ood_images)
    np.save(os.path.join(base_path, 'ood_split_cl.npy'), np.hstack((ood_concepts, ood_labels.reshape(-1,1))))
    print('Dataset saved!')
    print('Train:', len(train_images))
    print('Val:', len(val_images))
    print('Test:', len(test_images))
    print('OOD:', len(ood_images))

# TODO: Remove it
class ImbalancedDatasetSampler(Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices)

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):  # Note: for single attribute dataset
        return dataset.data[idx]['attribute_label'][0]

    def __iter__(self):
        idx = (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))
        return idx

    def __len__(self):
        return self.num_samples