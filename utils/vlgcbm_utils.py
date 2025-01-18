from torchvision import models
from torch.utils.data import DataLoader, Dataset, TensorDataset
from pytorchcv.model_provider import get_model as ptcv_get_model
from datasets import get_dataset
from typing import Optional, Tuple
import torch
import utils.clip as clip
from utils.utils import get_resnet_imagenet_preprocess
import os
import json
import numpy as np
from loguru import logger
from functools import partial
from config import LABELS

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
        image, target = self.torch_dataset[idx]

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
        image, target = self.torch_dataset[idx]

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
        return concept_one_hot

    def _find_in_list(self, concept: str, bbxs):
        # randomly pick a bounding box
        matched_bbxs = [bbx for bbx in bbxs if concept == bbx["label"]]
        return len(matched_bbxs) > 0, matched_bbxs

    def _get_data(self, idx):
        data_file = f"{self.dir}/{idx}.json"
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
        image, target = self.torch_dataset[idx]
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
    else:
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