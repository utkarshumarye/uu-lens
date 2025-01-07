import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # Remove last classification layer

    def forward_one(self, x):
        return self.resnet(x)

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        return torch.abs(out1 - out2)

    def extract_features(self, x):
        return self.forward_one(x)

def train_siamese_model(train_loader):
    model = SiameseNetwork()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for (img1, img2), labels in train_loader:
            optimizer.zero_grad()
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), './model_files/siamese_similarity.pth')
    print("Siamese Network Training completed.")
