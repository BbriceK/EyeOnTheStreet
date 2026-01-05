import os
import json
import random
import numpy as np
import torch

def seed_everything(seed=1111):
    """
    Set seeds for reproducibility across random, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_labels_from_json(json_file: str) -> dict:
    """
    Load labels from a JSON file.

    Args:
        json_file (str): Path to the JSON file containing labels.

    Returns:
        dict: Dictionary mapping image/file identifiers to labels.
    """
    with open(json_file, "r") as f:
        return json.load(f)

def save_embeddings(save_dir, split, cls, reg, patch, labels):
    """
    Save computed embeddings and labels as NumPy files.

    Args:
        save_dir (str): Directory where files will be saved. Created if not exists.
        split (str): Dataset split name, e.g., 'train', 'val', 'test'.
        cls (np.ndarray): Class token embeddings.
        reg (np.ndarray): Registered token embeddings.
        patch (np.ndarray): Patch token embeddings.
        labels (np.ndarray): Labels corresponding to the images.
    """
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"{split}_cls_embeddings.npy"), cls)
    np.save(os.path.join(save_dir, f"{split}_reg_embeddings.npy"), reg)
    np.save(os.path.join(save_dir, f"{split}_patch_embeddings.npy"), patch)
    np.save(os.path.join(save_dir, f"y_{split}.npy"), labels)
