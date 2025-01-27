"""Custom Dataset for Buoy Data"""
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from torchvision import datapoints
import torch
import yaml
import os
import numpy as np
import cv2
import random
import warnings

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from src.core import register

__all__ = ['BuoyDataset']

@register
def collate_fn_adapt(batch):
    img, queries, labels, queries_mask, labels_mask, target = zip(*batch)
    img = torch.stack(img, dim=0)
    pad_q = pad_sequence(queries, batch_first=True, padding_value = 0.0)
    pad_l = pad_sequence(labels, batch_first=True, padding_value = 0.0)
    pad_mask_q = pad_sequence(queries_mask, batch_first=True, padding_value=False)
    pad_mask_l = pad_sequence(labels_mask, batch_first=True, padding_value=False)

    return img, pad_q, pad_l, pad_mask_q, pad_mask_l, target
    

@register
class BuoyDataset(Dataset):
    def __init__(self, yaml_file, mode='train', transform=False, augment=False) -> None:
        # mode: train/test/val
        super().__init__()

        self.yaml_file = yaml_file
        if mode in ["train", "test", "val"]:
            self.mode = mode
        else:
            raise ValueError(f"Invalid mode ({mode}) for DataSet")
        self.data_path = None

        tf = transforms.Compose([
            transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet stats
                                    std=[0.229, 0.224, 0.225])
        ])
        self.transform = None
        if transform:
            self.transform = tf
        self.augment = augment
        self.augment_thresh = 0.5
        random.seed(0)
        torch.manual_seed(0)
        self.processYAML()

        self.labels = sorted(os.listdir(os.path.join(self.data_path, "labels")))
        self.images = sorted(os.listdir(os.path.join(self.data_path, "images")))
        self.queries = sorted(os.listdir(os.path.join(self.data_path, "queries")))

        self.checkdataset()

    def processYAML(self):
        if not os.path.exists(self.yaml_file):
            raise ValueError(f"Path to Dataset not found - Incorrect YAML File Path: {self.yaml_file}")
        with open(self.yaml_file, 'r') as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
            if self.mode in data:
                self.data_path = data[self.mode]
                if not os.path.exists(self.data_path):
                    raise ValueError(f"Incorrect path to {self.mode} folder in YAML file: {self.data_path}")
            else:
                raise ValueError(f"YAML file does not contain path to {self.mode} folder")

    def checkdataset(self):
        for label, image, query in zip(self.labels, self.images, self.queries):
            if not image.split(".")[0] == label.split('.')[0] == query.split('.')[0]:
                print(f"Warning, file not matching: {label}, {image}, {query}")

    def __len__(self):
        return len(self.labels)

    def flip_img(self, img, labels, queries):
        # flips image horizontally 
        img = cv2.flip(img, 1)
        if labels.numel() > 0:
            labels[:,1] = 1 - labels[:,1]
        queries[:,-1] *= -1
        return img, labels, queries

    def queries_add_noise(self, queries, dist_coeff=15, bearing_coeff=10):
        # adds noise to queries (dist, bearing) to simulate inaccuracy in chart data
        noise = lambda: 2 * (torch.rand(queries.size(dim=0), dtype=torch.float32) - 0.5)
        delta_dist = noise() * dist_coeff
        delta_bearing = torch.atan2(noise() * bearing_coeff, queries[:, 1]) / torch.pi * 180
        queries[:, 1] += delta_dist
        queries[:, 2] += delta_bearing
        return queries

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.data_path, "images", self.images[index]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        labels = torch.tensor(np.loadtxt(os.path.join(self.data_path, 'labels', self.labels[index])), dtype=torch.float32)
        queries = torch.tensor(np.loadtxt(os.path.join(self.data_path, 'queries', self.queries[index])),
                               dtype=torch.float32)[..., 0:3] # only take the first three datapoints in the label file (id, dist, angle)

        # ensure 2D shape:
        if queries.ndim == 1:
            queries = queries.unsqueeze(0)
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)

        if self.augment and random.random() > self.augment_thresh:
            img, labels, queries = self.flip_img(img, labels, queries)
            queries = self.queries_add_noise(queries)

        # normalize query inputs (dist and angle)
        queries[..., 1] = queries[..., 1] / 1000 # normalize dist between 0-1 (gets clamped later to max 1)
        queries[..., 2] = queries[..., 2] / 180 # normalize angle between -1 to 1

        labels_extended = torch.zeros(queries.size(dim=0), 5, dtype=torch.float32)
        if labels.numel() > 0:
            labels_extended[labels[:, 0].int(), :] = labels[:, :]

        labels_mask = torch.full((labels_extended.size(dim=0),), fill_value=False)
        if labels.numel() > 0:
            labels_mask[labels[:, 0].int()] = True

        queries_mask = torch.full((queries.size(dim=0),), fill_value=True)

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img).permute(2, 0, 1) / 255

        name = os.path.join(self.data_path, "images", self.images[index])

        # add target dict for rt-detr loss
        target = {}
        _,h,w = img.size()
        if labels.numel() > 0:
            labels_target = torch.zeros((labels.size(0), 4), dtype=torch.float32)
            labels_target[...,0] = (w *(labels[...,1] - labels[..., 3] / 2)).clamp(0, w)
            labels_target[...,1] = (h *(labels[...,2] - labels[..., 4] / 2)).clamp(0, h)
            labels_target[...,2] = (w *(labels[...,1] + labels[..., 3] / 2)).clamp(0, w)
            labels_target[...,3] = (h *(labels[...,2] + labels[..., 4] / 2)).clamp(0, h)
            target['boxes'] = datapoints.BoundingBox(
                labels_target.tolist(), 
                format=datapoints.BoundingBoxFormat.XYXY, 
                spatial_size=(h, w) # h w
            )
            target["labels"] = torch.ones((labels.size(0)), dtype=torch.float32)
        else:
            target["boxes"] = torch.empty((0, 4))
            target["labels"] = torch.empty((0,))
        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        target["size"] = torch.as_tensor([int(w), int(h)])
        
        sample = (img, queries, labels_extended, queries_mask, labels_mask, target)

        return sample
        
