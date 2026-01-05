import torch
import torch.nn as nn

class PatchCNN(nn.Module):
    """
    CNN module to process patch embeddings into a fixed-size global feature.

    Args:
        in_channels (int): Number of input channels (embedding dimension per patch)
        out_dim (int): Output feature dimension after CNN

    Architecture:
        - Two 3x3 conv layers with BatchNorm and ReLU
        - AdaptiveAvgPool2d(1) to pool spatial features into a single vector per sample
    """
    def __init__(self, in_channels, out_dim):
        super(PatchCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return x

class NeckDINov2(nn.Module):
    """
    Neck module for DINOv2 embeddings to combine CLS, REG, and Patch embeddings
    into a single feature vector for downstream tasks.

    Args:
        dropout (float): dropout rate for regularization
    """
    def __init__(self, dropout=0.7):
        super().__init__()
        self.conv = nn.Conv1d(384, 384, kernel_size=4, stride=4)
        self.patch_cnn = PatchCNN(in_channels=384, out_dim=128)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        reg_embeddings, cls_embeddings, patch_embeddings = x

        # Remove unnecessary singleton dimensions
        cls_embeddings = cls_embeddings.squeeze(1)
        reg_embeddings = reg_embeddings.squeeze(1)
        patch_embeddings = patch_embeddings.squeeze(1)

        # 1D conv on reg embeddings (sequence -> features)
        feature = self.drop(self.conv(reg_embeddings.transpose(1,2))).squeeze(-1)
        # Reshape patch embeddings to (B, C, H, W) for CNN
        B = patch_embeddings.size(0)
        patch_reshaped = patch_embeddings.view(B,16,16,384).permute(0,3,1,2)
        global_embedding = self.patch_cnn(patch_reshaped)

        # Concatenate REG features, CLS features, and patch global embedding
        return torch.cat((feature, cls_embeddings, global_embedding), dim=-1)
