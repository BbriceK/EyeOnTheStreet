import torch
import torch.nn as nn
from .patch_cnn import NeckDINov2

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        
        self.dropout1 = nn.Dropout(0.7)
        self.dropout2 = nn.Dropout(0.7)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.dim()==3: x = x.squeeze(1)
        x = self.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        
        x = self.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        return self.fc3(x)

class FullModel(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.feature_extractor = NeckDINov2(nc=8)
        self.classifier = Classifier(input_dim=896, output_dim=output_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

