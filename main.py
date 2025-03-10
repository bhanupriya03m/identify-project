import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np

from cnn import create_cnn
from pnn import PNN

#  Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#  Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load train, test, and val datasets
train_dataset = datasets.ImageFolder(root=r"D:\Final_Year_Project\Animal_footprint_detection\data\cropped_imgs\train", transform=transform)
test_dataset = datasets.ImageFolder(root=r"D:\Final_Year_Project\Animal_footprint_detection\data\cropped_imgs\test", transform=transform)
val_dataset = datasets.ImageFolder(root=r"D:\Final_Year_Project\Animal_footprint_detection\data\cropped_imgs\val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Combine all class names
class_names = list(set(train_dataset.classes + test_dataset.classes + val_dataset.classes))
class_names.sort()
num_classes = len(class_names)

print(f"Total classes found across train, test, and val: {num_classes}")
print(f"Class names: {class_names}")

# Initialize CNN
cnn = create_cnn(num_classes, feature_extraction=False).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)

# Train the CNN
cnn.train()
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

# Save trained CNN
torch.save(cnn.state_dict(), "cnn_model.pth")
print("CNN model saved as cnn_model.pth")

# Convert CNN to feature extractor mode
cnn = create_cnn(num_classes, feature_extraction=True).to(device)
cnn.load_state_dict(torch.load("cnn_model.pth", map_location=device), strict=False)

# Extract Features
features = []
labels = []

cnn.eval()
with torch.no_grad():
    for images, label in train_loader:
        images = images.to(device)
        output = cnn(images).cpu()
        features.extend(output.numpy())
        labels.extend(label.numpy())

# Train PNN
pnn = PNN()
pnn.fit(features, labels)

# Save CNN + PNN Feature Set
torch.save({
    'cnn': cnn.state_dict(),
    'pnn': pnn,
    'features': features,
    'labels': labels,
    'class_names': class_names
}, "cnn_pnn_model.pth")

print("CNN + PNN features saved as cnn_pnn_model.pth")
