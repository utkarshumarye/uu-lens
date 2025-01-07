import torch
import torch.nn as nn
from torchvision import models

class ResNetSimilarity(nn.Module):
    def __init__(self):
        super(ResNetSimilarity, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # Removing the final classification layer

    def forward(self, x):
        return self.resnet(x).view(x.size(0), -1)  # Flatten the output

    def extract_features(self, x):
        with torch.no_grad():
            return self.forward(x)
