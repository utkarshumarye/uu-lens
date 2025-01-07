import torch
import torch.nn as nn
import torch.optim as optim

class DeepHashingSimilarity(nn.Module):
    def __init__(self):
        super(DeepHashingSimilarity, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 32)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    def extract_features(self, x):
        with torch.no_grad():
            return self.forward(x)

def train_deep_hashing(train_loader):
    model = DeepHashingSimilarity()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), './model_files/deep_hashing_similarity.pth')
    print("Deep Hashing Model Training completed.")
