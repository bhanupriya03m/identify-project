import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from cnn import create_cnn
from utils.train import train_cnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load dataset
train_dataset = datasets.ImageFolder(root=r"D:\Final_Year_Project\Animal_footprint_detection\data\OpenAnimalTracks\cropped_imgs\train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Initialize CNN
num_classes = len(train_dataset.classes)
model = create_cnn(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train CNN
train_cnn(model, train_loader, criterion, optimizer, device, epochs=10)

# Save CNN checkpoint
torch.save(model.state_dict(), "cnn_model.pth")
print("CNN model saved as cnn_model.pth")
