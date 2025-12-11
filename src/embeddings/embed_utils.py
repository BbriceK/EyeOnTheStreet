import os
import json
import random
import numpy as np
import torch

def seed_everything(seed=1111):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_labels_from_json(json_file: str) -> dict:
    with open(json_file, "r") as f:
        return json.load(f)

def save_embeddings(save_dir, split, cls, reg, patch, labels):
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"{split}_cls_embeddings.npy"), cls)
    np.save(os.path.join(save_dir, f"{split}_reg_embeddings.npy"), reg)
    np.save(os.path.join(save_dir, f"{split}_patch_embeddings.npy"), patch)
    np.save(os.path.join(save_dir, f"y_{split}.npy"), labels)
