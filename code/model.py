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


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=0.5, gamma_pos=0.3, clip=0.01, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class PatchCNN(nn.Module):
    def __init__(self, in_channels, out_dim):
        super(PatchCNN, self).__init__()
        # s
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, out_dim, kernel_size=3, padding=1), 
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1),  
        )

    def forward(self, x):  
        x = self.cnn(x)          
        x = x.view(x.size(0), -1)            
        return x


class NeckDINov2(nn.Module):
    def __init__(self, nc, dropout=0.7):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=384, out_channels=384, kernel_size=4, stride=4)
        self.patch_cnn = PatchCNN(in_channels=384, out_dim=128)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        reg_embeddings, cls_embeddings,patch_embeddings = x

        cls_embeddings = cls_embeddings.squeeze(1)
        reg_embeddings = reg_embeddings.squeeze(1)
        patch_embeddings = patch_embeddings.squeeze(1)

        feature = self.drop(self.conv(reg_embeddings.transpose(1, 2))).squeeze(-1)
        B = patch_embeddings.size(0)

        patch_reshaped = patch_embeddings.view(B, 16, 16, 384).permute(0, 3, 1, 2)
        global_embedding = self.patch_cnn(patch_reshaped)
        x0 = torch.cat((feature, cls_embeddings, global_embedding), dim=-1)
        return x0


class FullModel(nn.Module):
    def __init__(self, output_dim):
        super(FullModel, self).__init__()
        self.feature_extractor = NeckDINov2(nc=8)
        self.classifier = Classifier(input_dim=896, output_dim=output_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output

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


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)

        self.dropout1 = nn.Dropout(0.7)
        self.dropout2 = nn.Dropout(0.7)

        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()


    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)

        x = self.relu(self.fc1(x))
        x = self.batch_norm1(x)
        x = self.dropout1(x)

        x = self.relu(self.fc2(x))
        x = self.batch_norm2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x


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
