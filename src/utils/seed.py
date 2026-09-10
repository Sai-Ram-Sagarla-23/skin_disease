"""
Reproducibility utilities.

Every script in this project calls set_seed(cfg['project']['seed']) before
doing anything else that involves randomness (splitting, augmentation,
weight initialization, dataloader shuffling).
"""
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch (CPU + all GPUs) for reproducibility.

    Note: cudnn.deterministic=True can slow down training slightly; this is
    an intentional trade-off in favor of reproducible research results.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Detect and report the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Device: CUDA")
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_mem_gb:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Device: CPU")
        print("GPU: Not available (training will be slower; reduce batch_size "
              "and image_size in configs/config.yaml if needed)")
    return device
