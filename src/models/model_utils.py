import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

def seed_everything(seed=1111):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_embeddings(emb_path):
    def load_split(split):
        return (
            np.load(os.path.join(emb_path, f"{split}_cls_embeddings.npy"), allow_pickle=True).item(),
            np.load(os.path.join(emb_path, f"{split}_reg_embeddings.npy"), allow_pickle=True).item(),
            np.load(os.path.join(emb_path, f"{split}_patch_embeddings.npy"), allow_pickle=True).item(),
            np.load(os.path.join(emb_path, f"y_{split}.npy"), allow_pickle=True),
        )

    return {
        "train": load_split("train"),
        "val": load_split("val"),
        "test": load_split("test"),
    }


class EmbeddingDataset(Dataset):
    def __init__(self, cls_embeddings, reg_embeddings, patch_embeddings, labels, image_names=None):
        # Convert dict values to stacked torch tensors
        self.cls_embeddings = torch.from_numpy(
            np.stack(list(cls_embeddings.values()))
        ).float()

        self.reg_embeddings = torch.from_numpy(
            np.stack(list(reg_embeddings.values()))
        ).float()

        self.patch_embeddings = torch.from_numpy(
            np.stack(list(patch_embeddings.values()))
        ).float()

        self.labels = torch.from_numpy(labels).float()
        self.image_names = image_names

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.image_names is not None:
            return (
                self.reg_embeddings[idx],
                self.cls_embeddings[idx],
                self.patch_embeddings[idx],
            ), self.labels[idx], self.image_names[idx]
        else:
            return (
                self.reg_embeddings[idx],
                self.cls_embeddings[idx],
                self.patch_embeddings[idx],
            ), self.labels[idx]


def build_loaders(data, batch_size):
    train_ds = EmbeddingDataset(*data["train"])
    val_ds   = EmbeddingDataset(*data["val"])
    test_ds  = EmbeddingDataset(*data["test"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train() 
    running_loss = 0.0
    num_batches = len(loader)

    for (reg, cls, patch), labels in loader:
        reg, cls, patch, labels = reg.to(device), cls.to(device), patch.to(device), labels.to(device)

        optimizer.zero_grad()       
        outputs = model((reg, cls, patch))  
        loss = criterion(outputs, labels)   
        loss.backward()            
        optimizer.step()      

        running_loss += loss.item() 

    avg_loss = running_loss / num_batches
    return avg_loss


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    num_batches = len(loader)
    y_pred, y_true = [], []

    with torch.no_grad():
        for (reg, cls, patch), labels in loader:
            reg, cls, patch, labels = reg.to(device), cls.to(device), patch.to(device), labels.to(device)

            outputs = model((reg, cls, patch))
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            probs = torch.sigmoid(outputs)
            y_pred.append(probs.detach().cpu())
            y_true.append(labels.detach().cpu())

    # Concatenate all batches
    y_pred = torch.cat(y_pred, dim=0).numpy()
    y_true = torch.cat(y_true, dim=0).numpy()
    avg_loss = running_loss / num_batches

    return avg_loss, y_pred, y_true
