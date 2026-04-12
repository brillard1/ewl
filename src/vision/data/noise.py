"""
Label noise injection for noise-robustness ablation (Prediction 1 from paper).

Usage:
    dataset = inject_label_noise(dataset, noise_ratio=0.2, num_classes=100, seed=42)
"""

import numpy as np
from torch.utils.data import Dataset


class NoisyLabelDataset(Dataset):
    """Wraps any vision dataset and randomly flips a fraction of labels."""

    def __init__(self, dataset, noise_ratio, num_classes, seed=42):
        """
        Args:
            dataset:     Original dataset (must return dict with "labels" key).
            noise_ratio: Fraction of labels to corrupt, e.g. 0.2 = 20%.
            num_classes: Number of classes (for uniform random replacement).
            seed:        RNG seed for reproducibility.
        """
        assert 0.0 <= noise_ratio < 1.0
        self.dataset = dataset
        self.noise_ratio = noise_ratio

        rng = np.random.default_rng(seed)
        n = len(dataset)
        n_noisy = int(n * noise_ratio)

        noisy_indices = rng.choice(n, size=n_noisy, replace=False)
        self.noisy_set = set(noisy_indices.tolist())

        # Pre-generate corrupted labels: uniform random over all classes except true label
        self.corrupted_labels = {}
        for idx in noisy_indices:
            true_label = dataset[idx]["labels"]
            choices = [c for c in range(num_classes) if c != true_label]
            self.corrupted_labels[int(idx)] = int(rng.choice(choices))

        print(f"NoisyLabelDataset: {n_noisy}/{n} labels corrupted ({noise_ratio*100:.0f}%)")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if idx in self.corrupted_labels:
            sample = dict(sample)  # don't mutate original
            sample["labels"] = self.corrupted_labels[idx]
        return sample


def inject_label_noise(dataset, noise_ratio, num_classes, seed=42):
    """
    Returns the dataset unchanged if noise_ratio == 0, else wraps it.
    """
    if noise_ratio == 0.0:
        return dataset
    return NoisyLabelDataset(dataset, noise_ratio, num_classes, seed)
