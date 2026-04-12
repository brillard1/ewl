"""
CUB-200-2011 Dataset for Vision Tasks
"""

import os
import urllib.request
import tarfile
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CUB200Dataset(Dataset):
    """
    CUB-200-2011 Dataset for fine-grained bird classification

    - 200 bird species
    - 11,788 images (5,994 train / 5,794 test)

    Args:
        root: Root directory to store dataset
        train: If True, use training split; else test split
        transform: Optional transform to apply to images
        download: If True, download dataset if not present
    """

    def __init__(self, root, train=True, transform=None, download=True):
        self.root = os.path.join(root, "CUB_200_2011")
        self.train = train
        self.transform = transform

        if download and not os.path.exists(self.root):
            self._download()

        # Load image paths and labels
        images_df = pd.read_csv(
            os.path.join(self.root, "images.txt"),
            sep=" ",
            header=None,
            names=["img_id", "filepath"],
        )
        labels_df = pd.read_csv(
            os.path.join(self.root, "image_class_labels.txt"),
            sep=" ",
            header=None,
            names=["img_id", "label"],
        )
        train_test_df = pd.read_csv(
            os.path.join(self.root, "train_test_split.txt"),
            sep=" ",
            header=None,
            names=["img_id", "is_train"],
        )

        # Merge and filter by split
        data = images_df.merge(labels_df, on="img_id").merge(train_test_df, on="img_id")
        data = data[data["is_train"] == (1 if train else 0)]

        self.images = data["filepath"].values
        self.labels = (data["label"].values - 1).astype(int)  # 0-indexed

        print(f"[CUB200] Loaded {len(self)} {'train' if train else 'test'} samples")

    def _download(self):
        """Download and extract CUB-200-2011 dataset"""
        url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
        os.makedirs(os.path.dirname(self.root), exist_ok=True)
        tar_path = os.path.join(os.path.dirname(self.root), "CUB_200_2011.tgz")

        print(f"[CUB200] Downloading dataset...")
        urllib.request.urlretrieve(url, tar_path)

        print(f"[CUB200] Extracting...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(os.path.dirname(self.root))

        os.remove(tar_path)
        print(f"[CUB200] Download complete!")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns:
            dict with 'pixel_values', 'labels', and 'sample_id'
        """
        img_path = os.path.join(self.root, "images", self.images[idx])
        img = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return {"pixel_values": img, "labels": label, "sample_id": idx}

    @property
    def num_classes(self):
        return 200


def get_cub_transforms(image_size=224, augmentation=True):
    """
    Get image transforms for ViT following standard fine-tuning practice

    Args:
        image_size: Input image size (default 224)
        augmentation: Whether to apply data augmentation

    Returns:
        transform: Composed transforms

    Standard ViT fine-tuning pipeline:
    - Training: RandomResizedCrop for scale/position variation
    - Validation: Resize(256) + CenterCrop(224) for consistency
    """
    if augmentation:
        # Training: RandomResizedCrop provides scale diversity
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        # Validation: Resize + CenterCrop (standard practice)
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    return transform
