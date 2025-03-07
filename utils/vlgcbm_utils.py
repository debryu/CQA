from torchvision import models
from torch.utils.data import DataLoader, Dataset, TensorDataset
from pytorchcv.model_provider import get_model as ptcv_get_model
import sys
from datasets import get_dataset
from typing import Optional, Tuple
import torch
import wandb
from tqdm import tqdm
import utils.clip as clip
from utils.utils import get_resnet_imagenet_preprocess
import os
import json
import numpy as np
from loguru import logger
from functools import partial
from config import LABELS
from models.glm_saga.elasticnet import IndexedTensorDataset, glm_saga

BACKBONE_ENCODING_DIMENSION = {
    "resnet18_cub": 512,
    "clip_RN50": 1024,
    "clip_RN50_penultimate": 2048,
    "resnet50": 2048,
}

'''
Avoid anonymous functions in the code. They can't be pickled and will cause issues when saving the model.
These two functions are created to avoid the issue.
'''
def target_model_function(model, x):
    return model.encode_image(x).float()
'''---------------------------------'''

def get_target_model(target_name, device):
    if target_name.startswith("clip_"):
        target_name = target_name[5:]
        model, preprocess = clip.load(target_name, device=device)
        target_model = partial(target_model_function, model)

    elif target_name == "resnet18_places":
        target_model = models.resnet18(pretrained=False, num_classes=365).to(device)
        state_dict = torch.load("data/resnet18_places365.pth.tar")["state_dict"]
        new_state_dict = {}
        for key in state_dict:
            if key.startswith("module."):
                new_state_dict[key[7:]] = state_dict[key]
        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()

    elif target_name == "resnet18_cub":
        target_model = ptcv_get_model("resnet18_cub", pretrained=True).to(device)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()

    elif target_name.endswith("_v2"):
        target_name = target_name[:-3]
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V2".format(target_name_cap))
        target_model = eval("models.{}(weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()

    else:
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()

    return target_model, preprocess

def format_concept(s):
    # replace - with ' '
    # replace , with ' '
    # only one space between words
    s = s.lower()
    s = s.replace("-", " ")
    s = s.replace(",", " ")
    s = s.replace(".", " ")
    s = s.replace("(", " ")
    s = s.replace(")", " ")
    if s[:2] == "a ":
        s = s[2:]
    elif s[:3] == "an ":
        s = s[3:]

    # remove trailing and leading spaces
    s = " ".join(s.split())
    return s

def get_concepts(concept_file: str, filter_file:Optional[str]=None):
    with open(concept_file) as f:
        concepts = f.read().split("\n")

    # remove repeated concepts and maintain order
    concepts = list(dict.fromkeys([format_concept(concept) for concept in concepts]))

    # check for filter file
    if filter_file and os.path.exists(filter_file):
        logger.info(f"Filtering concepts using {filter_file}")
        with open(filter_file) as f:
            to_filter_concepts = f.read().split("\n")
        to_filter_concepts = [format_concept(concept) for concept in to_filter_concepts]
        concepts = [concept for concept in concepts if concept not in to_filter_concepts]

    logger.debug(concepts)
    return concepts

def get_classes(dataset_name):
    return LABELS[dataset_name]

def load_concept_and_count(
    save_dir: str, file_name: str = "concept_counts.txt", filter_file:Optional[str]=None
):
    with open(os.path.join(save_dir, file_name), "r") as f:
        lines = f.readlines()
        concepts = []
        counts = []
        for line in lines:
            concept = line.split(" ")[:-1]
            concept = " ".join(concept)
            count = line.split(" ")[-1]
            concepts.append(format_concept(concept))
            counts.append(float(count))

    if filter_file and os.path.exists(filter_file):
        with open(filter_file) as f:
            logger.info(f"Filtering concepts using {filter_file}")
            to_filter_concepts = f.read().split("\n")
        to_filter_concepts = [format_concept(concept) for concept in to_filter_concepts]
        counts = [count for concept, count in zip(concepts, counts) if concept not in to_filter_concepts]
        concepts = [concept for concept in concepts if concept not in to_filter_concepts]
        assert len(concepts) == len(counts)

    return concepts, counts

def get_filtered_concepts_and_counts(
    dataset_name,
    raw_concepts,
    preprocess=None,
    val_split: Optional[float] = 0.1,
    batch_size: int = 256,
    num_workers: int = 4,
    confidence_threshold: float = 0.10,
    label_dir="outputs",
    use_allones: bool = False,
    seed: int = 42,
    remove_never_seen=False
):
    # remove concepts that are not present in the dataset
    dataloader = get_concept_dataloader(
        dataset_name,
        "train",
        raw_concepts,
        preprocess=preprocess,
        val_split=val_split,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        confidence_threshold=confidence_threshold,
        crop_to_concept_prob=0.0,
        label_dir=label_dir,
        use_allones=use_allones,
        seed=seed,
        concept_only=True
    )
    # get concept counts
    raw_concepts_count = torch.zeros(len(raw_concepts))
    for data in tqdm(dataloader):
        raw_concepts_count += data[1].sum(dim=0)
    
    print(raw_concepts[1])
    print(raw_concepts[51])
    logger.debug(f"Filtered concepts index: {torch.where(raw_concepts_count==0)}")
    # remove concepts that are not present in the dataset
    
    raw_concepts_count = raw_concepts_count.numpy()
    if remove_never_seen:
        concepts = [concept for concept, count in zip(raw_concepts, raw_concepts_count) if count > 0]
        concept_counts = [count for _, count in zip(raw_concepts, raw_concepts_count) if count > 0]
        filtered_concepts = [concept for concept, count in zip(raw_concepts, raw_concepts_count) if count == 0]
    else:
        concepts = [concept for concept, count in zip(raw_concepts, raw_concepts_count)]
        concept_counts = [count for _, count in zip(raw_concepts, raw_concepts_count)]
        filtered_concepts = [concept for concept, count in zip(raw_concepts, raw_concepts_count)]
    print(f"Filtered {len(raw_concepts) - len(concepts)} concepts")

    return concepts, concept_counts, filtered_concepts

def save_concept_count(
    concepts,
    counts,
    save_dir: str,
    file_name: str = "concept_counts.txt",
):
    with open(os.path.join(save_dir, file_name), "w") as f:
        if len(concepts) != len(counts):
            raise ValueError("Length of concepts and counts should be the same")
        f.write(f"{concepts[0]} {counts[0]}")
        for concept, count in zip(concepts[1:], counts[1:]):
            f.write(f"\n{concept} {count}")

def save_filtered_concepts(
    filtered_concepts,
    save_dir: str,
    file_name: str = "filtered_concepts.txt",
):
    with open(os.path.join(save_dir, file_name), "w") as f:
        if len(filtered_concepts) > 0:
            f.write(filtered_concepts[0])
            for concept in filtered_concepts[1:]:
                f.write("\n" + concept)

class Backbone(torch.nn.Module):
    # store intermediate feature values from backbone
    feature_vals = {}

    def __init__(self, backbone_name: str, feature_layer: str, device: str = "cuda"):
        super().__init__()
        target_model, target_preprocess = get_target_model(
            backbone_name, device
        )

        # hook into feature layer
        def hook(module, input, output):
            self.feature_vals[output.device] = output

        command = "target_model.{}.register_forward_hook(hook)".format(feature_layer)
        eval(command)

        # assign backbone and preprocess
        self.backbone = target_model
        self.preprocess = target_preprocess
        self.output_dim = BACKBONE_ENCODING_DIMENSION[backbone_name]

    def forward(self, x):
        out = self.backbone(x)
        return self.feature_vals[out.device].mean(dim=[2, 3])

    def save_model(self, save_dir):
        torch.save(self.backbone.state_dict(), os.path.join(save_dir, "backbone.pt"))

    @classmethod
    def from_pretrained(cls, load_path: str, device: str = "cuda"):
        # load args
        model = cls.from_args(load_path, device)
        model.backbone.load_state_dict(
            torch.load(os.path.join(load_path, "backbone.pt"))
        )
        return model

    @classmethod
    def from_args(cls, load_dir: str, device: str = "cuda"):
        with open(os.path.join(load_dir, "args.txt"), "r") as f:
            args = json.load(f)
        return cls(args["backbone"], args["feature_layer"], device)
    
class BackboneCLIP(torch.nn.Module):
    def __init__(
        self, backbone_name: str, use_penultimate: bool = True, device: str = "cuda"
    ):
        super().__init__()
        target_model, target_preprocess = clip.load(backbone_name[5:], device=device)
        if use_penultimate:
            logger.info("Using penultimate layer of CLIP")
            target_model = target_model.visual
            N = target_model.attnpool.c_proj.in_features
            identity = torch.nn.Linear(N, N, dtype=torch.float16, device=device)
            torch.nn.init.zeros_(identity.bias)
            identity.weight.data.copy_(torch.eye(N))
            target_model.attnpool.c_proj = identity
            self.output_dim = BACKBONE_ENCODING_DIMENSION[
                f"{backbone_name}_penultimate"
            ]
        else:
            logger.info("Using final layer of CLIP")
            target_model = target_model.visual
            self.output_dim = BACKBONE_ENCODING_DIMENSION[backbone_name]

        # assign backbone and preprocess
        self.backbone = target_model.float()
        self.preprocess = target_preprocess

    def forward(self, x):
        output = self.backbone(x).float()
        return output

    def save_model(self, save_dir):
        torch.save(self.backbone.state_dict(), os.path.join(save_dir, "backbone.pt"))

    @classmethod
    def from_args(cls, load_dir: str, device: str = "cuda"):
        with open(os.path.join(load_dir, "args.txt"), "r") as f:
            args = json.load(f)
        return cls(args["backbone"], args["use_clip_penultimate"], device)

    @classmethod
    def from_pretrained(cls, load_path: str, device: str = "cuda"):
        # load args
        model = cls.from_args(load_path, device)
        model.backbone.load_state_dict(
            torch.load(os.path.join(load_path, "backbone.pt"))
        )
        return model
    
class ConceptLayer(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_hidden: int = 0,
        bias: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        model = [torch.nn.Linear(in_features, out_features, bias=bias)]
        for _ in range(num_hidden):
            model.append(torch.nn.ReLU())
            model.append(torch.nn.Linear(out_features, out_features, bias=bias))

        self.model = torch.nn.Sequential(*model).to(device)
        self.out_features = out_features
        logger.info(self.model)

    def forward(self, x):
        return self.model(x)

    def save_model(self, save_dir):
        # save model
        torch.save(self.state_dict(), os.path.join(save_dir, "cbl.pt"))

    @classmethod
    def from_pretrained(cls, load_path: str, device: str = "cuda"):
        # load args
        with open(os.path.join(load_path, "args.txt"), "r") as f:
            args = json.load(f)

        num_hidden = args["cbl_hidden_layers"]
        if args["use_clip_penultimate"] and args["backbone"].startswith("clip"):
            encoder_dim = BACKBONE_ENCODING_DIMENSION[
                f"{args['backbone']}_penultimate"
            ]
        else:
            encoder_dim = BACKBONE_ENCODING_DIMENSION[args["backbone"]]
        num_concepts = len(get_concepts(f"{load_path}/concepts.txt"))

        # create model
        model = cls(encoder_dim, num_concepts, num_hidden=num_hidden, device=device)
        model.load_state_dict(torch.load(os.path.join(load_path, "cbl.pt")))
        return model
    
class NormalizationLayer(torch.nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, device: str = "cuda"):
        super().__init__()
        self.mean = mean.to(device)
        self.std = std.to(device)

    def forward(self, x):
        return (x - self.mean) / self.std

    def save_model(self, save_dir):
        # save model
        torch.save(self.mean, os.path.join(save_dir, "train_concept_features_mean.pt"))
        torch.save(self.std, os.path.join(save_dir, "train_concept_features_std.pt"))

    @classmethod
    def from_pretrained(cls, load_path: str, device: str = "cuda"):
        # load args
        with open(os.path.join(load_path, "args.txt"), "r") as f:
            args = json.load(f)

        mean = torch.load(
            os.path.join(load_path, "train_concept_features_mean.pt"),
            map_location=device,
        )
        std = torch.load(
            os.path.join(load_path, "train_concept_features_std.pt"),
            map_location=device,
        )
        normalization_layer = cls(mean, std, device=device)
        return normalization_layer
    
class FinalLayer(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, device: str = "cuda"):
        super().__init__(in_features, out_features, bias=True)
        self.to(device)

    def forward(self, x):
        return super().forward(x)

    def save_model(self, save_dir):
        # save model
        torch.save(self.state_dict(), os.path.join(save_dir, "final.pt"))

    @classmethod
    def from_pretrained(cls, load_path: str, device: str = "cuda"):
        # load args
        with open(os.path.join(load_path, "args.txt"), "r") as f:
            args = json.load(f)

        num_concepts = len(get_concepts(f"{load_path}/concepts.txt"))
        num_classes = len(get_classes(args["dataset"]))

        # create model
        model = cls(num_concepts, num_classes, device=device)
        model.load_state_dict(torch.load(os.path.join(load_path, "final.pt")))
        return model

''' 
--------------------------------------------------- 
UTILS from VLG-CBM/data/concept_dataset.py 
--------------------------------------------------- 
'''

def get_bbox_iou(boxA, boxB):
    # Source: https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

class ConceptDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        torch_dataset: Dataset,
        concepts = None,
        split_suffix="train",
        label_dir: str = "outputs",
        confidence_threshold: float = 0.10,
        preprocess=None,
        crop_to_concept_prob: bool = 0.0,
        overlap_iou_threshold: float = 0.5,
        concept_only=False
    ):
        self.torch_dataset = torch_dataset
        self.concepts = concepts
        self.dir = f"{label_dir}/{dataset_name}_{split_suffix}"
        self.confidence_threshold = confidence_threshold
        self.preprocess = preprocess
        self.overlap_iou_threshold = overlap_iou_threshold
        self.concept_only = concept_only
        # Return cropped image containing a single concept
        # with probability `crop_to_concept_prob`
        self.crop_to_concept_prob = crop_to_concept_prob

    def __len__(self):
        return len(self.torch_dataset)

    def __getitem__(self, idx):
        if self.concept_only:
            return 0, self._get_concept(idx), 0 # 0 is placeholder
        prob = np.random.rand()
        if prob < self.crop_to_concept_prob:
            try:
                return self.__getitem__per_concept(idx)
            except Exception as e:
                logger.warning(f"Failed to get item {idx} per concept: {e}")

        return self.__getitem__all(idx)

    def __getitem__per_concept(self, idx):
        image, _, target = self.torch_dataset[idx]

        # return 1 hot vector of concepts
        data = self._get_data(idx)

        bbxs = data[1:]
        bbxs = [bbx for bbx in bbxs if bbx["logit"] > self.confidence_threshold]
        for bbx in bbxs:
            bbx["label"] = format_concept(bbx["label"])

        # get mapping of concepts to a random bounding box containing the concept
        concept_bbx_map = []
        for concept_idx, concept in enumerate(self.concepts):
            _, matched_bbxs = self._find_in_list(concept, bbxs)
            if len(matched_bbxs) > 0:
                concept_bbx_map.append((concept_idx, matched_bbxs[np.random.randint(0, len(matched_bbxs))]))

        # get one hot vector of concepts
        concept_one_hot = torch.zeros(len(self.concepts), dtype=torch.float)
        if len(concept_bbx_map) > 0:
            # randomly pick a concept and its bounding box
            random_concept_idx, random_bbx = concept_bbx_map[np.random.randint(0, len(concept_bbx_map))]
            concept_one_hot[random_concept_idx] = 1.0
            image = image.crop(random_bbx["box"])

            # mark concepts with high overlap with the selected concept as 1
            for bbx in bbxs:
                if bbx["label"] == random_bbx["label"]:
                    continue
                else:
                    iou = get_bbox_iou(random_bbx["box"], bbx["box"])
                    try:
                        if iou > self.overlap_iou_threshold:
                            concept_idx = self.concepts.index(bbx["label"])
                            concept_one_hot[concept_idx] = 1.0
                            # logger.debug(f"Marking {bbx['concept']} as 1 due to overlap with {random_bbx['concept']}")
                    except ValueError:
                        continue

        # preprocess image
        if self.preprocess:
            image = self.preprocess(image)

        return image, concept_one_hot, target

    def __getitem__all(self, idx):
        image, _, target = self.torch_dataset[idx]

        # get raw data
        data = self._get_data(idx)

        # get one hot vector of concepts
        bbxs = data[1:]
        bbxs = [bbx for bbx in bbxs if bbx["logit"] > self.confidence_threshold]
        for bbx in bbxs:
            bbx["label"] = format_concept(bbx["label"])
        # get one hot vector of concepts
        concept_one_hot = [1 if self._find_in_list(concept, bbxs)[0] else 0 for concept in self.concepts]
        concept_one_hot = torch.tensor(concept_one_hot, dtype=torch.float)

        # preprocess image
        if self.preprocess:
            image = self.preprocess(image)

        return image, concept_one_hot, target
    
    def _get_concept(self, idx):
        # return 1 hot vector of concepts
        data = self._get_data(idx)

        # get one hot vector of concepts
        bbxs = data[1:]
        bbxs = [bbx for bbx in bbxs if bbx["logit"] > self.confidence_threshold]
        for bbx in bbxs:
            bbx["label"] = format_concept(bbx["label"])
        
        # get one hot vector of concepts
        concept_one_hot = [1 if self._find_in_list(concept, bbxs)[0] else 0 for concept in self.concepts]
        concept_one_hot = torch.tensor(concept_one_hot, dtype=torch.float)
        #print(concept_one_hot)
        return concept_one_hot

    def _find_in_list(self, concept: str, bbxs):
        #for bb in bbxs:
        #   if bb["label"].strip().replace(" ","_") != concept.strip().replace(" ","_"):
        #        if 'hooked' in concept.strip().replace(" ","_") or 'length' in concept.strip().replace(" ","_"):
        #            if 'hooked' in bb["label"].strip().replace(" ","_") or 'length' in bb["label"].strip().replace(" ","_"):
        #                print(bb["label"].strip().replace(" ","_"), "- C:", concept.strip().replace(" ","_"))
                #if concept.strip().replace(" ","_") == 'hooked seabird bill shape' or concept.strip().replace(" ","_") == 'bill length about the same as head':
                    
        #            if bb["label"].strip().replace(" ","_") != concept.strip().replace(" ","_"):
        #                print(bb["label"].strip().replace(" ","_"), "- C:", concept.strip().replace(" ","_"))
        # randomly pick a bounding box
        # .replace(" _","_").replace(" _","_")
        matched_bbxs = [bbx for bbx in bbxs if concept.strip().replace(" ","_") == bbx["label"].strip().replace(" ","_")]
        return len(matched_bbxs) > 0, matched_bbxs

    def _get_data(self, idx):
        data_file = f"{self.dir}/{idx}.json"
        try:
            with open(data_file, "r") as f:
                data = json.load(f)
        except:
            logger.error(f"Missing {idx}.json")
            data_file = f"{self.dir}/0.json"
            with open(data_file, "r") as f:
                data = json.load(f)
        return data

    def get_annotations(self, idx: int):
        return self._get_data(idx)[1:]
    
    ''' asdgteyt34tdfs
    def visualize_annotations(self, idx: int):
        image_pil = self.torch_dataset[idx][0]
        annotations = self._get_data(idx)[1:]
        fig = plot_annotations(image_pil, annotations)
        fig.show()

    def plot_annotations(self, idx: int, annotations: List[Dict[str, Any]]):
        image_pil = self.torch_dataset[idx][0]
        fig = plot_annotations(image_pil, annotations)
        fig.show()
    '''
    def get_image_pil(self, idx: int):
        return self.torch_dataset[idx][0]

    def get_target(self, idx):
        _, target = self.torch_dataset[idx]
        return target    

class AllOneConceptDataset(ConceptDataset):
    def __init__(self, classes, *args, **kwargs):
        print(args, kwargs)
        super().__init__(*args, **kwargs)
        self.per_class_concepts = len(self.concepts) // len(classes)
        logger.info(f"Assigning {self.per_class_concepts} concepts to each class")

    def __getitem__(self, idx):
        image, _, target = self.torch_dataset[idx]
        if self.preprocess:
            image = self.preprocess(image)
        concept_one_hot = torch.zeros((len(self.concepts),), dtype=torch.float)
        concept_one_hot[target * self.per_class_concepts : (target + 1) * self.per_class_concepts] = 1
        return image, concept_one_hot, target

def get_concept_dataloader(
    dataset_name: str,
    split: str,
    concepts,
    preprocess=None,
    val_split: Optional[float] = 0.1,
    batch_size: int = 256,
    num_workers: int = 4,
    shuffle: bool = False,
    confidence_threshold: float = 0.10,
    crop_to_concept_prob: float = 0.0,
    label_dir="outputs",
    use_allones=False,
    seed: int = 42,
    concept_only=False
):
    dataset = ConceptDataset if not use_allones else partial(AllOneConceptDataset, get_classes(dataset_name))
    if split == "test":
        dataset = dataset(
            dataset_name,
            get_dataset(dataset_name, split="test"),
            concepts,
            split_suffix="val",
            preprocess=preprocess,
            confidence_threshold=confidence_threshold,
            crop_to_concept_prob=crop_to_concept_prob,
            label_dir=label_dir,
            concept_only=concept_only
        )
        logger.info(f"Test dataset size: {len(dataset)}")
    elif split == 'train':
        dataset = dataset(
            dataset_name,
            get_dataset(dataset_name, split="train"),
            concepts,
            split_suffix="train",
            preprocess=preprocess,
            confidence_threshold=confidence_threshold,
            crop_to_concept_prob=crop_to_concept_prob,
            label_dir=label_dir,
            concept_only=concept_only
        )
    elif split == 'val' or split == 'valid':
        dataset = dataset(
            dataset_name,
            get_dataset(dataset_name, split="val"),
            concepts,
            split_suffix="val",
            preprocess=preprocess,
            confidence_threshold=confidence_threshold,
            crop_to_concept_prob=crop_to_concept_prob,
            label_dir=label_dir,
            concept_only=concept_only
        )
    else:
        raise NotImplementedError
        assert val_split is not None
        dataset = dataset(
            dataset_name,
            get_dataset(dataset_name, split="train"),
            concepts,
            split_suffix="train",
            preprocess=preprocess,
            confidence_threshold=confidence_threshold,
            crop_to_concept_prob=crop_to_concept_prob,
            label_dir=label_dir,
            concept_only=concept_only
        )

        # get split indices
        n_val = int(val_split * len(dataset))
        n_train = len(dataset) - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
        )  # ensure same split in same run

        if split == "train":
            logger.info(f"Train dataset size: {len(train_dataset)}")
            dataset = train_dataset
        else:
            logger.info(f"Val dataset size: {len(val_dataset)}")
            dataset = val_dataset

    if num_workers > 1:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    else:
        print("Using single worker")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader#, dataset

def get_final_layer_dataset(
    backbone: Backbone,
    cbl: ConceptLayer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    save_dir: str,
    load_dir: str = None,
    batch_size: int = 256,
    device="cuda",
    filter=None,
):
    if load_dir is None:
        logger.info("Creating final layer training and validation datasets")
        with torch.no_grad():
            train_concept_features = []
            train_concept_labels = []
            logger.info("Creating final layer training dataset")
            for features, _, labels in tqdm(train_loader):  #Originally was for features, _, labels in tqdm(train_loader):
                #print(features.shape)
                #print(c.shape)
                #print(labels.shape)
                #asd
                features = features.to(device)
                concept_logits = cbl(backbone(features))
                train_concept_features.append(concept_logits.detach().cpu())
                train_concept_labels.append(labels)
            train_concept_features = torch.cat(train_concept_features, dim=0)
            train_concept_labels = torch.cat(train_concept_labels, dim=0)

            val_concept_features = []
            val_concept_labels = []
            logger.info("Creating final layer validation dataset")
            for features, _, labels in tqdm(val_loader):
                features = features.to(device)
                concept_logits = cbl(backbone(features))
                val_concept_features.append(concept_logits.detach().cpu())
                val_concept_labels.append(labels)
            val_concept_features = torch.cat(val_concept_features, dim=0)
            val_concept_labels = torch.cat(val_concept_labels, dim=0)

            # normalize concept features
            train_concept_features_mean = train_concept_features.mean(dim=0)
            train_concept_features_std = train_concept_features.std(dim=0)
            train_concept_features = (train_concept_features - train_concept_features_mean) / train_concept_features_std
            val_concept_features = (val_concept_features - train_concept_features_mean) / train_concept_features_std

            # normalization layer
            normalization_layer = NormalizationLayer(train_concept_features_mean, train_concept_features_std, device=device)
    else:
        # load normalized concept features
        logger.info("Loading final layer training dataset")
        train_concept_features = torch.load(os.path.join(load_dir, "train_concept_features.pt"))
        train_concept_labels = torch.load(os.path.join(load_dir, "train_concept_labels.pt"))
        val_concept_features = torch.load(os.path.join(load_dir, "val_concept_features.pt"))
        val_concept_labels = torch.load(os.path.join(load_dir, "val_concept_labels.pt"))
        normalization_layer = NormalizationLayer.from_pretrained(load_dir, device=device)

    # save normalized concept features
    torch.save(train_concept_features, os.path.join(save_dir, "train_concept_features.pt"))
    torch.save(train_concept_labels, os.path.join(save_dir, "train_concept_labels.pt"))
    torch.save(val_concept_features, os.path.join(save_dir, "val_concept_features.pt"))
    torch.save(val_concept_labels, os.path.join(save_dir, "val_concept_labels.pt"))

    # save normalized concept features mean and std
    normalization_layer.save_model(save_dir)
    if filter is not None:
        train_concept_features = train_concept_features[:, filter]
        val_concept_features = val_concept_features[:, filter]
    # Note: glm saga expects y to be on CPU
    train_concept_dataset = IndexedTensorDataset(train_concept_features, train_concept_labels)
    val_concept_dataset = TensorDataset(val_concept_features, val_concept_labels)
    logger.info("Train concept dataset size: {}".format(len(train_concept_dataset)))
    logger.info("Val concept dataset size: {}".format(len(val_concept_dataset)))

    train_concept_loader = DataLoader(train_concept_dataset, batch_size=batch_size, shuffle=True)
    val_concept_loader = DataLoader(val_concept_dataset, batch_size=batch_size, shuffle=False)
    return train_concept_loader, val_concept_loader, normalization_layer


'''---------------------------------------------------------------------------------'''
'''
             LOSS FUNCTION UTILS
'''
'''---------------------------------------------------------------------------------'''

nINF = -100

class TwoWayLoss(torch.nn.Module):
    def __init__(self, Tp=4., Tn=1.):
        super(TwoWayLoss, self).__init__()
        self.Tp = Tp
        self.Tn = Tn

        logger.info(f"Initializing TwoWayLoss with Tp: {Tp} and Tn: {Tn}")

    def forward(self, x, y):
        class_mask = (y > 0).any(dim=0)
        sample_mask = (y > 0).any(dim=1)

        # Calculate hard positive/negative logits
        pmask = y.masked_fill(y <= 0, nINF).masked_fill(y > 0, float(0.0))
        plogit_class = torch.logsumexp(-x/self.Tp + pmask, dim=0).mul(self.Tp)[class_mask]
        plogit_sample = torch.logsumexp(-x/self.Tp + pmask, dim=1).mul(self.Tp)[sample_mask]
    
        nmask = y.masked_fill(y != 0, nINF).masked_fill(y == 0, float(0.0))
        nlogit_class = torch.logsumexp(x/self.Tn + nmask, dim=0).mul(self.Tn)[class_mask]
        nlogit_sample = torch.logsumexp(x/self.Tn + nmask, dim=1).mul(self.Tn)[sample_mask]

        return torch.nn.functional.softplus(nlogit_class + plogit_class).mean() + \
                torch.nn.functional.softplus(nlogit_sample + plogit_sample).mean()


def get_loss(type: str, num_concepts: int, num_samples:int, concept_counts, cbl_pos_weight:float, cbl_auto_weight: bool=False, tp: float = 4.,device="cuda"):
    if type == "bce":
        logger.info("Using BCE Loss for training CBL...")
        if cbl_auto_weight:
            logger.info(f"Using automatic weighting for positive examples with scale {cbl_pos_weight}")
            pos_count = torch.tensor(concept_counts).to(device)
            neg_count = num_samples - pos_count
            scale = (neg_count / pos_count) * cbl_pos_weight
            logger.info(f"scale mean: {scale.mean()}, scale std: {scale.std()}")
            logger.info(f"scale min: {scale.min()}, scale max: {scale.max()}")
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=scale).to(device)
        else:
            logger.info("Using fixed weighting for positive examples")
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cbl_pos_weight] * num_concepts)).to(device)
    elif type == "twoway":
        logger.info("Using TwoWay Loss for training CBL...")
        loss_fn = TwoWayLoss(Tp=tp)
    else:
        raise NotImplementedError(f"Loss {type} is not implemented")
    

    return loss_fn

'''---------------------------------------------------------------------------------'''
'''
             TRAIN UTILS
'''
'''---------------------------------------------------------------------------------'''

def validate_cbl(
    backbone: Backbone,
    cbl: ConceptLayer,
    val_loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: str = "cuda",
    args = None,
):
    val_loss = 0.0
    with torch.no_grad():
        logger.info("Running CBL validation")
        for features, concept_one_hot, _ in tqdm(val_loader):
            features = features.to(device)
            concept_one_hot = concept_one_hot.to(device)

            # forward pass
            concept_logits = cbl(backbone(features))

            # calculate loss
            batch_loss = loss_fn(concept_logits, concept_one_hot)
            val_loss += batch_loss.item()
            if args.mock:
                break
        # finalize metrics and update model
        val_loss = val_loss / len(val_loader)

    return val_loss

def train_cbl(
    backbone: Backbone,
    cbl: ConceptLayer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    loss_fn: torch.nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    concepts = None,    # List[str]
    tb_writer=None,
    device: str = "cuda",
    finetune: bool = False,
    optimizer: str = "sgd",
    scheduler: str = None,
    backbone_lr: float = 1e-3,
    data_parallel=False,
    args = None,
):
    # setup optimizer
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(
            cbl.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
        )
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(cbl.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError
    if finetune:
        optimizer.add_param_group({"params": backbone.parameters(), "lr": backbone_lr})

    # setup schedular
    if scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_loss = float("inf")
    best_val_loss_epoch = None
    best_model_state = None
    if data_parallel:
        backbone = torch.nn.DataParallel(backbone)
        cbl = torch.nn.DataParallel(cbl)
    
    for epoch in range(epochs):
        logs = {'epoch':epoch}
        train_loss = 0
        lr = optimizer.param_groups[0]["lr"]

        logger.info(f"Running CBL training for Epoch: {epoch}")
        its = tqdm(total=len(train_loader), position=0, leave=True)
        train_losses = []
        for batch_idx, (features, concept_one_hot, _) in enumerate(train_loader):
            features = features.to(device)  # (batch_size, feature_dim)
            concept_one_hot = concept_one_hot.to(device)  # (batch_size, n_concepts)

            # forward pass
            if finetune:
                backbone.train()
                embeddings = backbone(features)
            else:
                with torch.no_grad():
                    embeddings = backbone(features)  # (batch_size, feature_dim)
            concept_logits = cbl(embeddings)  # (batch_size, n_concepts)

            # calculate loss
            batch_loss = loss_fn(concept_logits, concept_one_hot)
            train_loss += batch_loss.item()

            # backprop
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            train_losses.append(batch_loss.item())
            # print batch stats
            if (batch_idx + 1) % 1000 == 0:
                its.update(1000)
                print(
                    "Epoch: {} | Batch: {} | Loss: {:.6f}".format(
                        epoch, batch_idx, batch_loss.item()
                    )
                )

                # exit if loss is nan
                if torch.isnan(batch_loss):
                    # Exit process if loss is nan
                    logger.error(f"Loss is nan at epoch {epoch} and batch {batch_idx}")
                    sys.exit(1)
        backbone.eval()
        # finalize metrics and update model
        its.close()
        train_loss = train_loss / len(train_loader)
        # train_per_concept_roc = train_per_concept_roc.compute()

        # evaluate on validation set
        logger.info(f"Running CBL validation for Epoch: {epoch}")
        val_loss = validate_cbl(
            backbone,
            cbl.module if data_parallel else cbl,
            val_loader,
            loss_fn=loss_fn,
            device=device,
            args=args,
        )
        if val_loss < best_val_loss:
            logger.info(f"Updating best val loss at epoch: {epoch}")
            best_val_loss = val_loss
            best_val_loss_epoch = epoch
            best_backbone_state = backbone.state_dict()
            best_model_state = cbl.state_dict()
        logs['val_loss'] = val_loss
        logs['train_loss'] = np.mean(train_losses)
        # write to tensorboard
        if tb_writer is not None:
            tb_writer.add_scalar("Loss/train", train_loss, epoch)
            tb_writer.add_scalar("Loss/val", val_loss, epoch)
            tb_writer.add_scalar("lr", lr, epoch)

        if args.wandb:
            wandb.log(logs)
        # print epoch stats
        logger.info(
            f"Epoch: {epoch} | Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}"
        )

        # Step the scheduler
        if scheduler is not None:
            scheduler.step(val_loss)
        
    

    # return best model based on validation loss
    logger.info(f"Best val loss: {best_val_loss:.6f} at epoch {best_val_loss_epoch}")
    cbl.load_state_dict(best_model_state)
    backbone.load_state_dict(best_backbone_state)
    if data_parallel:
        cbl = cbl.module
        backbone = backbone.module
    return cbl, backbone

def test_model(
    loader: DataLoader,
    backbone: Backbone,
    cbl: ConceptLayer,
    normalization: NormalizationLayer,
    final_layer: FinalLayer,
    device: str = "cuda",
    args = None,
):
    acc_mean = 0.0
    for features, concept_one_hot, targets in tqdm(loader):
        features = features.to(device)
        concept_one_hot = concept_one_hot.to(device)
        targets = targets.to(device)

        # forward pass
        with torch.no_grad():
            embeddings = backbone(features)
            concept_logits = cbl(embeddings)
            concept_probs = normalization(concept_logits)
            logits = final_layer(concept_probs)
            if args.mock:
                break

        # calculate accuracy
        preds = logits.argmax(dim=1)
        accuracy = (preds == targets).sum().item()
        acc_mean += accuracy

    return acc_mean / len(loader.dataset)

def train_sparse_final(
    linear,
    indexed_train_loader,
    val_loader,
    n_iters,
    lam,
    step_size=0.1,
    device="cuda",
):
    # zero initialize
    num_classes = linear.weight.shape[0]
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    ALPHA = 0.99
    metadata = {}
    metadata["max_reg"] = {}
    metadata["max_reg"]["nongrouped"] = lam

    # Solve the GLM path
    output_proj = glm_saga(
        linear,
        indexed_train_loader,
        step_size,
        n_iters,
        ALPHA,
        epsilon=1,
        k=1,
        val_loader=val_loader,
        do_zero=False,
        metadata=metadata,
        n_ex=len(indexed_train_loader.dataset),
        n_classes=num_classes,
        verbose=True,
    )

    return output_proj

def per_class_accuracy(
    model: torch.nn.Module, loader: DataLoader, classes, device: str = "cuda"
): #Classes: List[str], output -> Dict[str, float]
    correct = torch.zeros(len(classes)).to(device)
    total = torch.zeros(len(classes)).to(device)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for features, _, targets in tqdm(loader):
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            preds = logits.argmax(dim=1)
            for pred, target in zip(preds, targets):
                total[target] += 1
                if pred == target:
                    correct[target] += 1

    per_class_accuracy = correct / total
    total_accuracy = correct.sum() / total.sum()
    total_datapoints = total.sum()

    # return a dictionary of class names and accuracies, and total accuracy
    return {
        "Per class accuracy": {
            classes[i]: f"{per_class_accuracy[i]*100.0:.2f}"
            for i in range(len(classes))
        },
        "Overall accuracy": f"{total_accuracy*100.0:.2f}",
        "Datapoints": f"{total_datapoints}",
    }