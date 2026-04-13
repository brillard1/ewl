"""
FGVC-Aircraft Dataset for Vision Tasks

100 aircraft variant classes, 10,000 images total.
Official splits: train (3334), val (3333), test (3333).
We combine all splits and re-split 80/10/10 in the pipeline.

Dataset structure after download:
    fgvc-aircraft-2013b/data/
        images/                       # {image_id}.jpg
        variants.txt                  # 100 class names, one per line
        images_variant_train.txt      # "{image_id} {variant name}" per line
        images_variant_val.txt
        images_variant_test.txt
"""

import os
import urllib.request
import tarfile
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
from torchvision import transforms


class AircraftDataset(Dataset):
    """
    FGVC-Aircraft variant classification dataset.

    Args:
        root:      Root directory (dataset saved under root/fgvc-aircraft-2013b/).
        split:     One of "train", "validation", "test".
        transform: Optional transform applied to PIL images.
        download:  If True, download dataset if not present.
    """

    URL = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
    NUM_CLASSES = 100

    def __init__(self, root="./data", split="train", transform=None, download=True):
        assert split in ("train", "val", "test"), \
            f"split must be train/val/test, got {split!r}"

        self.root = os.path.join(root, "fgvc-aircraft-2013b", "data")
        self.split = split
        self.transform = transform

        if download and not os.path.exists(self.root):
            self._download(root)

        # Build class → index mapping from variants.txt
        variants_path = os.path.join(self.root, "variants.txt")
        with open(variants_path) as f:
            variants = [line.strip() for line in f if line.strip()]
        self.class_to_idx = {v: i for i, v in enumerate(variants)}

        # Load image list for this split
        ann_path = os.path.join(self.root, f"images_variant_{split}.txt")
        image_ids, label_list = [], []
        with open(ann_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # format: "{image_id} {variant name}"  (variant may contain spaces)
                parts = line.split(" ", 1)
                image_id = parts[0]
                variant = parts[1] if len(parts) > 1 else ""
                image_ids.append(image_id)
                label_list.append(self.class_to_idx[variant])

        self.image_ids = image_ids
        self.labels = np.array(label_list, dtype=np.int64)

        print(f"[Aircraft] Loaded {len(self)} {split} samples ({self.NUM_CLASSES} classes)")

    def _download(self, root):
        os.makedirs(root, exist_ok=True)
        tar_path = os.path.join(root, "fgvc-aircraft-2013b.tar.gz")
        print("[Aircraft] Downloading dataset (2.75 GB)...")
        urllib.request.urlretrieve(self.URL, tar_path)
        print("[Aircraft] Extracting...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(root)
        os.remove(tar_path)
        print("[Aircraft] Download complete.")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        """Returns dict with 'pixel_values' (PIL or tensor), 'labels', 'sample_id'."""
        img_path = os.path.join(self.root, "images", f"{self.image_ids[idx]}.jpg")
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return {"pixel_values": img, "labels": int(self.labels[idx]), "sample_id": idx}

    @property
    def num_classes(self):
        return self.NUM_CLASSES


def get_aircraft_transforms(image_size=224, augmentation=True):
    """
    Image transforms for FGVC-Aircraft + ViT.

    Fine-grained aircraft recognition benefits from tighter crops
    (less background) so we use scale=(0.5, 1.0) for training.
    """
    if augmentation:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
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
