import torch
import torch.nn as nn
from torchvision import models

class VisionTransformerSimilarity(nn.Module):
    def __init__(self):
        super(VisionTransformerSimilarity, self).__init__()
        vit = models.vit_b_16(pretrained=True)
        self.vit = vit

    def forward(self, x):
        return self.vit(x).last_hidden_state.mean(dim=1)

    def extract_features(self, x):
        with torch.no_grad():
            return self.forward(x)
