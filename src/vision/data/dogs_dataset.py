"""
Stanford Dogs Dataset for Vision Tasks

120 dog breeds, 20,580 images.
Official splits: 12,000 train / 8,580 test (defined by lists/train_list.mat and test_list.mat).
No official validation split — pipeline does stratified 80/20 split of train.

Dataset page: http://vision.stanford.edu/aditya86/ImageNetDogs/
Download:
  images: http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
  lists:  http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar
"""

import os
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import scipy.io
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

_IMAGES_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
_LISTS_URL  = "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar"


class StanfordDogsDataset(Dataset):
    """
    Stanford Dogs Dataset.

    Returns dicts matching the project standard: {'pixel_values', 'labels', 'sample_id'}.

    Args:
        root:      Directory where 'StanfordDogs/' will be created.
        split:     'train' or 'test'.
        transform: Optional transform applied to PIL images.
        download:  Download dataset if not present.
    """

    def __init__(self, root="./data", split="train", transform=None, download=False):
        self.root = Path(root) / "StanfordDogs"
        self.split = split
        self.transform = transform
        self._num_classes = 120

        if download and not self._check_exists():
            self._download()

        if not self._check_exists():
            raise RuntimeError(
                "Stanford Dogs not found. Use download=True or download manually:\n"
                f"  images: {_IMAGES_URL}\n"
                f"  lists:  {_LISTS_URL}\n"
                f"Extract both tars into: {self.root}  (Images/ and *_list.mat files should be at the root)"
            )

        self._load_split(split)
        print(f"[StanfordDogs] Loaded {len(self)} {split} samples ({self._num_classes} classes)")

    def _check_exists(self):
        return (self.root / "Images").exists() and (self.root / "train_list.mat").exists()

    def _download(self):
        self.root.mkdir(parents=True, exist_ok=True)
        for url, name in [(_IMAGES_URL, "images.tar"), (_LISTS_URL, "lists.tar")]:
            dest = self.root / name
            print(f"[StanfordDogs] Downloading {name}...")
            urllib.request.urlretrieve(url, dest)
            print(f"[StanfordDogs] Extracting {name}...")
            with tarfile.open(dest) as tar:
                tar.extractall(self.root)
            dest.unlink()
        print("[StanfordDogs] Download complete.")

    def _load_split(self, split):
        mat_file = self.root / f"{split}_list.mat"
        mat = scipy.io.loadmat(mat_file)

        # file_list: array of arrays of shape (1,) containing strings like
        #   'n02085620-Chihuahua/n02085620_10074.jpg'
        file_list = mat["file_list"].squeeze()   # (N,) object array
        labels    = mat["labels"].squeeze()      # (N,) int array, 1-indexed

        self.image_paths = [
            self.root / "Images" / str(f.item())
            for f in file_list
        ]
        self.labels = (labels - 1).tolist()      # 0-indexed

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"pixel_values": img, "labels": self.labels[idx], "sample_id": idx}

    @property
    def num_classes(self):
        return self._num_classes


def get_dogs_transforms(image_size=224, augmentation=True):
    """Standard ViT transforms for Stanford Dogs."""
    if augmentation:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
