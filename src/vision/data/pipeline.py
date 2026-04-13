"""
Vision data pipeline for image classification tasks

All datasets are combined from their native splits and re-split into
80/10/10 (train/val/test) using a seeded random split.
"""

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from .cub_dataset import CUB200Dataset, get_cub_transforms
# Other dataset modules are imported lazily inside _load_full_dataset
# so that missing optional dataset files don't break the import


class NoisyLabelWrapper(Dataset):
    """Wraps a dataset and randomly flips `noise_rate` fraction of labels."""

    def __init__(self, dataset, num_classes, noise_rate, seed=42):
        self.dataset = dataset
        self.num_classes = num_classes
        self.noise_rate = noise_rate
        rng = np.random.RandomState(seed)
        n = len(dataset)
        self.noisy_mask = rng.rand(n) < noise_rate
        # For each noisy sample, pick a different class uniformly
        self.noisy_labels = rng.randint(0, num_classes, size=n)
        print(
            f"[DATA] NoisyLabelWrapper: flipping {self.noisy_mask.sum()} / {n} "
            f"labels ({noise_rate*100:.0f}% noise)"
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.noisy_mask[idx]:
            orig = item["labels"]
            new_label = int(self.noisy_labels[idx])
            # Ensure we don't accidentally keep the same label
            if new_label == orig:
                new_label = (new_label + 1) % self.num_classes
            return {**item, "labels": new_label}
        return item


class TransformedSubset(Dataset):
    """Wraps a Subset/Dataset and overrides the transform applied to images.

    The underlying dataset MUST be loaded with ``transform=None`` so that
    ``pixel_values`` is a raw PIL Image.  This wrapper applies *transform*
    in ``__getitem__`` and passes through all other fields unchanged.
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        # Propagate num_classes from the root dataset
        root = subset
        while hasattr(root, "dataset"):
            root = root.dataset
        self._num_classes = getattr(root, "_num_classes", None) or getattr(
            root, "num_classes", None
        )

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        item = self.subset[idx]
        img = item["pixel_values"]  # PIL Image (no transform was applied)
        if self.transform:
            img = self.transform(img)
        return {"pixel_values": img, "labels": item["labels"], "sample_id": idx}

    @property
    def num_classes(self):
        return self._num_classes


class _ReindexedSubset(Dataset):
    """
    Like torch.utils.data.Subset but reassigns sample_id to the local index
    [0, len(indices)) instead of inheriting the parent dataset's index.

    This keeps EMAWeighter's loss_ema tensor correctly sized: a bare Subset
    would pass parent-space indices into TransformedSubset.__getitem__, making
    sample_ids range up to N_train-1 even though EMAWeighter was allocated with
    the smaller subset size.
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.dataset[int(self.indices[idx])]
        item = dict(item)
        item["sample_id"] = idx  # reassign to local [0, subset_size) index
        return item


def _get_labels_fast(dataset):
    """Extract labels from any dataset without loading images.

    Recursively handles: raw datasets with .labels, torch.Subset, ConcatDataset.
    Falls back to iterating items only if none of the above apply.
    """
    # Raw dataset with a labels array (CUB200Dataset, AircraftDataset, etc.)
    if hasattr(dataset, "labels"):
        return np.array(dataset.labels)

    # torch.Subset — index parent labels by subset indices
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        parent = _get_labels_fast(dataset.dataset)
        return parent[np.array(dataset.indices)]

    # ConcatDataset
    if hasattr(dataset, "datasets"):
        return np.concatenate([_get_labels_fast(d) for d in dataset.datasets])

    # Slow fallback
    return np.array([dataset[i]["labels"] for i in range(len(dataset))])


def apply_class_imbalance(train_sub, imbalance_ratio, num_classes, seed=42):
    """Artificially create a long-tail class imbalance in the training split.

    Classes are sorted 0..C-1 and assigned an exponentially decaying sample budget:
        n_i = max(1, floor(n_max * imbalance_ratio^(i / (C-1))))
    so class 0 keeps all its samples and class C-1 retains ~imbalance_ratio * n_max.

    Args:
        train_sub: torch.Subset (output of random_split) before TransformedSubset
        imbalance_ratio: float in (0, 1]. Ratio of minority to majority class size.
        num_classes: total number of classes
        seed: random seed for reproducibility

    Returns:
        np.ndarray of selected local indices (into train_sub)
    """
    rng = np.random.RandomState(seed)
    labels = _get_labels_fast(train_sub)

    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
    counts = [len(ci) for ci in class_indices]
    n_max = max(counts) if counts else 1

    kept = []
    for c in range(num_classes):
        ci = class_indices[c]
        if len(ci) == 0:
            continue
        exponent = c / max(1, num_classes - 1)
        n_keep = max(1, int(n_max * (imbalance_ratio ** exponent)))
        n_keep = min(n_keep, len(ci))
        chosen = rng.choice(ci, n_keep, replace=False)
        kept.extend(chosen.tolist())

    kept = np.array(kept)
    total_kept = len(kept)
    minority_size = max(1, int(n_max * imbalance_ratio))
    print(
        f"[DATA] ClassImbalance: imbalance_ratio={imbalance_ratio}, "
        f"majority={n_max}, minority≈{minority_size}, retained {total_kept} samples"
    )
    return kept


def _load_official_splits(dataset_name, config):
    """Load official train/val/test splits for each dataset.

    Splitting strategy per dataset:
      CUB-200:       official train (5994) split 80/20 → train/val; official test (5794) used as-is.
                     Returns (train_ds, None, test_ds) — val=None signals caller to create val from train.
      Aircraft:      official train/val/test used directly (3334/3333/3333). No re-splitting.
      Stanford Dogs: official train/test; no official val → same as CUB (val from train).
      Food101:       official train/test; no official val → same as CUB (val from train).

    Returns:
        (train_ds, val_ds, test_ds)
        val_ds is None when the dataset has no official val split.
    """
    if dataset_name == "cub200":
        data_root = config["dataset"].get("data_root", "./data")
        train_ds = CUB200Dataset(root=data_root, train=True,  transform=None, download=True)
        test_ds  = CUB200Dataset(root=data_root, train=False, transform=None, download=True)
        train_ds._num_classes = test_ds._num_classes = 200
        return train_ds, None, test_ds  # val created from train in prepare_vision_dataset

    elif dataset_name == "aircraft":
        from .aircraft_dataset import AircraftDataset
        train_ds = AircraftDataset(split="train", transform=None)
        val_ds   = AircraftDataset(split="val",   transform=None)
        test_ds  = AircraftDataset(split="test",  transform=None)
        train_ds._num_classes = val_ds._num_classes = test_ds._num_classes = 100
        return train_ds, val_ds, test_ds

    elif dataset_name == "stanford_dogs":
        from .dogs_dataset import StanfordDogsDataset
        data_root = config["dataset"].get("data_root", "./data")
        train_ds = StanfordDogsDataset(root=data_root, split="train", transform=None, download=True)
        test_ds  = StanfordDogsDataset(root=data_root, split="test",  transform=None, download=True)
        train_ds._num_classes = test_ds._num_classes = 120
        return train_ds, None, test_ds

    elif dataset_name == "food101":
        from .food101_dataset import Food101Dataset
        data_root = config["dataset"].get("data_root", "./data")
        train_ds = Food101Dataset(root=data_root, split="train", transform=None, download=True)
        test_ds  = Food101Dataset(root=data_root, split="test",  transform=None, download=True)
        train_ds._num_classes = test_ds._num_classes = 101
        return train_ds, None, test_ds

    else:
        raise ValueError(f"Unknown vision dataset: {dataset_name}")


def _stratified_split(dataset, val_fraction=0.2, seed=42):
    """Split dataset into train/val preserving per-class proportions.

    For each class, takes floor(val_fraction * n_class) samples for val
    and the remainder for train. Returns two Subset objects.
    """
    rng = np.random.RandomState(seed)
    labels = _get_labels_fast(dataset)
    classes = np.unique(labels)

    train_indices, val_indices = [], []
    for c in classes:
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_fraction))
        val_indices.extend(idx[:n_val].tolist())
        train_indices.extend(idx[n_val:].tolist())

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def prepare_vision_dataset(config):
    """
    Prepare vision dataset using official splits where available.

    Splitting strategy:
      Aircraft:      official train / val / test (3334 / 3333 / 3333) — no re-splitting.
      CUB-200:       official train split 80/20 → train/val; official test used as-is.
      Stanford Dogs: official train split 80/20 → train/val; official test used as-is.
      Food101:       official train split 80/20 → train/val; official test used as-is.

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    dataset_name = config["dataset"]["name"]
    image_size = config["dataset"].get("image_size", 224)
    seed = config["dataset"]["seed"]

    # --- transforms -----------------------------------------------------------
    if dataset_name == "cub200":
        get_transforms = get_cub_transforms
    elif dataset_name == "aircraft":
        from .aircraft_dataset import get_aircraft_transforms
        get_transforms = get_aircraft_transforms
    elif dataset_name == "stanford_dogs":
        from .dogs_dataset import get_dogs_transforms
        get_transforms = get_dogs_transforms
    elif dataset_name == "food101":
        from .food101_dataset import get_food101_transforms
        get_transforms = get_food101_transforms
    else:
        raise ValueError(f"Unknown vision dataset: {dataset_name}")

    train_transform = get_transforms(
        image_size, augmentation=config["dataset"].get("augmentation", True)
    )
    eval_transform = get_transforms(image_size, augmentation=False)

    # --- load official splits -------------------------------------------------
    raw_train, raw_val, raw_test = _load_official_splits(dataset_name, config)
    num_classes = getattr(raw_train, "_num_classes", 200)

    if raw_val is not None:
        # Dataset has official val split (Aircraft) — use all three as-is
        train_sub = raw_train
        val_sub   = raw_val
        test_sub  = raw_test
        print(
            f"[DATA] {dataset_name}: official splits → "
            f"train {len(train_sub)} / val {len(val_sub)} / test {len(test_sub)}"
        )
    else:
        # No official val — stratified 80/20 split of official train by class
        train_sub, val_sub = _stratified_split(raw_train, val_fraction=0.2, seed=seed)
        test_sub = raw_test
        print(
            f"[DATA] {dataset_name}: stratified split of official train {len(raw_train)} → "
            f"train {len(train_sub)} / val {len(val_sub)}  (80/20)  |  official test {len(test_sub)}"
        )

    # --- optional class imbalance on train only -------------------------------
    imbalance_ratio = config["dataset"].get("imbalance_ratio", 0.0)
    if 0.0 < imbalance_ratio < 1.0:
        imb_indices = apply_class_imbalance(
            train_sub, imbalance_ratio, num_classes, seed=seed
        )
        train_dataset = _ReindexedSubset(
            TransformedSubset(train_sub, train_transform), imb_indices
        )
    else:
        train_dataset = TransformedSubset(train_sub, train_transform)

    val_dataset  = TransformedSubset(val_sub,  eval_transform)
    test_dataset = TransformedSubset(test_sub, eval_transform)

    # --- optional label noise on train only -----------------------------------
    noise_rate = config["dataset"].get("label_noise", 0.0)
    if noise_rate > 0.0:
        train_dataset = NoisyLabelWrapper(
            train_dataset, num_classes, noise_rate, seed=seed
        )

    return train_dataset, val_dataset, test_dataset


def create_vision_dataloaders(train_dataset, val_dataset, config, test_dataset=None):
    """
    Create vision dataloaders.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Configuration dictionary
        test_dataset: Optional test dataset

    Returns:
        train_loader, val_loader  (and test_loader if test_dataset given)
    """
    batch_size = config["training"].get("micro_batch_size", 64)
    num_workers = config["dataset"].get("num_workers", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        print(
            f"[DATA] Created 3 dataloaders: batch_size={batch_size}, num_workers={num_workers}"
        )
        return train_loader, val_loader, test_loader

    print(
        f"[DATA] Created vision dataloaders: batch_size={batch_size}, num_workers={num_workers}"
    )
    return train_loader, val_loader
