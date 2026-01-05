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

def load_labels_from_json(json_file: str, keep_labels=None) -> dict:
    """
    Load labels from a JSON file and optionally filter to specific categories.

    Args:
        json_file (str): Path to the JSON file containing labels.
        keep_labels (list[int], optional): List of label indices to keep. 
            If None, all labels are returned.

    Returns:
        dict: Dictionary mapping image/file identifiers to labels (filtered if keep_labels is given).
    """
    with open(json_file, "r") as f:
        labels_dict = json.load(f)

    if keep_labels is not None:
        labels_dict = {
            k: [v[i] for i in keep_labels]
            for k, v in labels_dict.items()
        }

    return labels_dict

def build_paths(data_root, labels_dict, split):
    """
    Build full filesystem paths for images belonging to a specific dataset split.

    Args:
        data_root (str): Root folder where images are stored.
        labels_dict (dict): Dictionary mapping image paths (relative to data_root) to labels.
        split (str): Dataset split to select ('train', 'val', or 'test').

    Returns:
        list[str]: List of full paths to images in the specified split.
    """
    return [os.path.join(data_root, k) for k in labels_dict if k.startswith(f"{split}/")]


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
