import torch
import torch.nn as nn
import torch.optim as optim

class AutoencoderSimilarity(nn.Module):
    def __init__(self):
        super(AutoencoderSimilarity, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.Sigmoid()  
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def extract_features(self, x):
        with torch.no_grad():
            return self.encoder(x).view(x.size(0), -1)

def train_autoencoder(train_loader):
    model = AutoencoderSimilarity()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for images, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), './model_files/autoencoder_similarity.pth')
    print("Autoencoder Training completed.")

def load_autoencoder():
    model = AutoencoderSimilarity()
    model.load_state_dict(torch.load('./model_files/autoencoder_similarity.pth'))
    model.eval()
    return model
