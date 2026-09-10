"""
Training-only augmentation. Kept to medically plausible transformations —
dermatoscopic images have no canonical "up" orientation, so flips/rotations
are safe, but we avoid unrealistic distortions (heavy shear, extreme color
jitter, cutout over the lesion, etc.) that could destroy diagnostic signal.
"""
from typing import Dict

from torchvision import transforms


def get_train_transform(cfg: Dict) -> transforms.Compose:
    size = cfg["data"]["image_size"]
    mean = cfg["data"]["mean"]
    std = cfg["data"]["std"]
    aug = cfg["data"]["augmentation"]

    return transforms.Compose([
        transforms.RandomResizedCrop(
            size, scale=tuple(aug["random_resized_crop_scale"])
        ),
        transforms.RandomHorizontalFlip(p=aug["horizontal_flip_p"]),
        transforms.RandomVerticalFlip(p=aug["vertical_flip_p"]),
        transforms.RandomRotation(degrees=aug["rotation_degrees"]),
        transforms.ColorJitter(
            brightness=aug["brightness"], contrast=aug["contrast"]
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
