"""
Food-101 Dataset for Vision Tasks

Official splits: 75,750 train / 25,250 test (750/250 per class, 101 classes).
No official validation split — pipeline does stratified 80/20 split of train.
"""

from pathlib import Path
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision.datasets as tv_datasets
from torch.utils.data import Dataset
from torchvision import transforms


class Food101Dataset(Dataset):
    """
    Wrapper around torchvision.datasets.Food101 that returns dicts matching
    the project's standard format: {'pixel_values', 'labels', 'sample_id'}.

    Args:
        root:      Root directory where the dataset is stored / will be downloaded.
        split:     'train' or 'test'.
        transform: Optional transform applied to PIL images.
        download:  Download if not present.
    """

    def __init__(self, root, split="train", transform=None, download=False):
        self._base = tv_datasets.Food101(
            root=root, split=split, transform=None, download=download
        )
        self.transform = transform
        self.labels = [label for label in self._base._labels]
        self._num_classes = 101
        print(f"[Food101] Loaded {len(self)} {split} samples ({self._num_classes} classes)")

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        img_path, label = self._base._image_files[idx], self._base._labels[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"pixel_values": img, "labels": label, "sample_id": idx}

    @property
    def num_classes(self):
        return self._num_classes


def get_food101_transforms(image_size=224, augmentation=True):
    """Standard ViT transforms for Food-101 (identical to CUB pipeline)."""
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
