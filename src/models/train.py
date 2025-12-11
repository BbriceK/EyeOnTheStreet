import argparse
import os
import shutil
import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision.transforms as T
from tqdm import tqdm
import json
from PIL import Image
import numpy as np
import random
from .full_model import FullModel
from .loss import AsymmetricLoss
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim as optim
import torch.nn.functional as F
from dinov2.models.vision_transformer import vit_small, vit_large
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix


def seed_everything(seed=1111):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_labels_from_json(json_file: str) -> dict:
    with open(json_file, "r") as f:
        labels_dict = json.load(f)
    return labels_dict


class EmbeddingDataset(Dataset):
    def __init__(self, cls_embeddings, reg_embeddings, patch_embeddings, labels, image_names=None):
        self.cls_embeddings = list(cls_embeddings.values())
        self.reg_embeddings = list(reg_embeddings.values())
        self.patch_embeddings = list(patch_embeddings.values())
        self.labels = labels
        self.image_names = image_names

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        cls_embedding = torch.tensor(self.cls_embeddings[idx], dtype=torch.float32)
        reg_embedding = torch.tensor(self.reg_embeddings[idx], dtype=torch.float32)
        patch_embedding = torch.tensor(self.patch_embeddings[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.image_names is not None:
            image_name = self.image_names[idx]
            return (reg_embedding, cls_embedding, patch_embedding), label, image_name
        else:
            return (reg_embedding, cls_embedding, patch_embedding), label



def main(emb_path, save_path, out_path, data_path, rank, local_rank, world_size, dist_url):
    seed_everything()

    dist.init_process_group(backend='nccl', init_method=dist_url, world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    labels_dict = load_labels_from_json(data_path)


    train_cls_embeddings = np.load(os.path.join(emb_path,"train_cls_embeddings.npy"), allow_pickle=True).item()
    train_reg_embeddings = np.load(os.path.join(emb_path,"train_reg_embeddings.npy"), allow_pickle=True).item()
    train_patch_embeddings = np.load(os.path.join(emb_path,"train_patch_embeddings.npy"), allow_pickle=True).item()
    y_train = np.load(os.path.join(emb_path,"y_train.npy"), allow_pickle=True)

    val_cls_embeddings = np.load(os.path.join(emb_path,"val_cls_embeddings.npy"), allow_pickle=True).item()
    val_reg_embeddings = np.load(os.path.join(emb_path,"val_reg_embeddings.npy"), allow_pickle=True).item()
    val_patch_embeddings = np.load(os.path.join(emb_path,"val_patch_embeddings.npy"), allow_pickle=True).item()
    y_val = np.load(os.path.join(emb_path,"y_val.npy"), allow_pickle=True)

    test_cls_embeddings = np.load(os.path.join(emb_path,"test_cls_embeddings.npy"), allow_pickle=True).item()
    test_reg_embeddings = np.load(os.path.join(emb_path,"test_reg_embeddings.npy"), allow_pickle=True).item()
    test_patch_embeddings = np.load(os.path.join(emb_path,"test_patch_embeddings.npy"), allow_pickle=True).item()
    y_test = np.load(os.path.join(emb_path,"y_test.npy"), allow_pickle=True)

    train_dataset = EmbeddingDataset(train_cls_embeddings, train_reg_embeddings, train_patch_embeddings, y_train)
    val_dataset = EmbeddingDataset(val_cls_embeddings, val_reg_embeddings, val_patch_embeddings, y_val)
    test_dataset = EmbeddingDataset(test_cls_embeddings, test_reg_embeddings, test_patch_embeddings, y_test)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)


    batch_size = 32
    num_classes = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)

    model = FullModel(8).cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)
    criterion = AsymmetricLoss()

    num_epochs = 150
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, ((reg_embeddings, cls_embeddings, patch_embeddings), labels) in enumerate(train_loader):
            reg_embeddings = reg_embeddings.to(device)
            cls_embeddings = cls_embeddings.to(device)
            patch_embeddings = patch_embeddings.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model((reg_embeddings, cls_embeddings, patch_embeddings))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        dist.all_reduce(torch.tensor([running_loss]).cuda(), op=dist.ReduceOp.SUM)
        avg_train_loss = running_loss / len(train_loader)
        dist.all_reduce(torch.tensor([avg_train_loss]).cuda(), op=dist.ReduceOp.SUM)
        avg_train_loss /= dist.get_world_size()

        if dist.get_rank() == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}")


        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch_idx, ((reg_embeddings, cls_embeddings, patch_embeddings), labels) in enumerate(val_loader):
                cls_embeddings = cls_embeddings.to(device)
                reg_embeddings = reg_embeddings.to(device)
                patch_embeddings = patch_embeddings.to(device)
                labels = labels.to(device)

                outputs = model((reg_embeddings, cls_embeddings, patch_embeddings))
                loss = criterion(outputs, labels)
                val_loss += loss.item()

            total_val_loss = torch.tensor(val_loss, dtype=torch.float32, device=device)
            dist.all_reduce(total_val_loss, op=dist.ReduceOp.SUM)
            total_val_loss /= dist.get_world_size()

            if dist.get_rank() == 0:
                avg_val_loss = total_val_loss.item() / len(val_loader)
                print(f"Validation Loss: {avg_val_loss:.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), save_path)
                    print(f"Best model saved at {save_path} with Validation Loss: {best_val_loss:.4f}")

    if dist.get_rank() == 0:
        print(f"Loading the best model from {save_path}")
        model.load_state_dict(torch.load(save_path))


    model.eval()
    test_loss = 0.0
    y_pred = []
    y_true = []

    with torch.no_grad():
        for batch_idx, ((reg_embeddings, cls_embeddings, patch_embeddings), labels) in enumerate(test_loader):
            cls_embeddings = cls_embeddings.to(device)
            reg_embeddings = reg_embeddings.to(device)
            patch_embeddings = patch_embeddings.to(device)
            labels = labels.to(device)

            outputs = model((reg_embeddings, cls_embeddings, patch_embeddings))
            loss = criterion(outputs, labels)

            probs = torch.sigmoid(outputs)
            y_pred.append(probs.cpu().detach().numpy())
            y_true.append(labels.cpu().detach().numpy())
            test_loss += loss.item()


    total_test_loss = torch.tensor(test_loss, dtype=torch.float32, device=device)
    dist.all_reduce(total_test_loss, op=dist.ReduceOp.SUM)
    total_test_loss /= dist.get_world_size()

    if dist.get_rank() == 0:
        avg_test_loss = total_test_loss.item() / len(test_loader)
        print(f"Testing Loss: {avg_test_loss:.4f}")

    y_pred_array = np.concatenate(y_pred, axis=0)
    y_true_array = np.concatenate(y_true, axis=0)

    y_pred_tensor = torch.tensor(y_pred_array, dtype=torch.float32, device=device)
    y_true_tensor = torch.tensor(y_true_array, dtype=torch.float32, device=device)

    gathered_preds = [torch.zeros_like(y_pred_tensor) for _ in range(dist.get_world_size())]
    gathered_trues = [torch.zeros_like(y_true_tensor) for _ in range(dist.get_world_size())]

    dist.all_gather(gathered_preds, y_pred_tensor)
    dist.all_gather(gathered_trues, y_true_tensor)

    y_pred_full = torch.cat(gathered_preds, dim=0).cpu().numpy()
    y_true_full = torch.cat(gathered_trues, dim=0).cpu().numpy()


    if dist.get_rank() == 0:
        np.save(os.path.join(out_path, "y_pred_full.npy"), y_pred_full)
        np.save(os.path.join(out_path, "y_true_full.npy"), y_true_full)
        print("Predictions and true labels saved as 'y_pred_full.npy' and 'y_true_full.npy'")

    torch.distributed.destroy_process_group()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('emb_pth', type=str, help='Path to the embedding folder')
    parser.add_argument('save_pth', type=str, help='Path to saving the weight')
    parser.add_argument('out_pth', type=str, help='Path to output files')
    parser.add_argument('dt_pth', type=str, help='Path to the dataset')
    parser.add_argument('--world_size', type=int, help='Total number of processes (GPUs)')
    parser.add_argument('--dist_url', help='URL for the distributed environment')
    args = parser.parse_args()
    rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ.get('SLURM_LOCALID'))
    main(args.emb_pth, args.save_pth, args.out_pth, args.dt_pth, rank, local_rank, args.world_size, args.dist_url)
