import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from cnn import create_cnn
from pnn import PNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load saved model
checkpoint = torch.load("cnn_pnn_model.pth")
cnn = create_cnn(18).to(device)
cnn.load_state_dict(checkpoint['cnn'])
cnn.fc = torch.nn.Identity()
cnn.eval()

features = checkpoint['features']
labels = checkpoint['labels']

pnn = PNN()
pnn.fit(features, labels)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Extract CNN features
    with torch.no_grad():
        feature = cnn(image).cpu().numpy().flatten()

    # Predict with PNN
    label_idx = pnn.predict(feature)

    # Load class names
    class_names = train_dataset.classes
    predicted_class = class_names[label_idx]

    print(f"Predicted Class: {predicted_class}")

# Test with an image
image_path = "data/test/footprint.jpg"
predict_image(image_path)
