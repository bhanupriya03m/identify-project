import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from cnn import create_cnn
from pnn import PNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Allow safe unpickling for NumPy
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

# ✅ Load saved model checkpoint (include features & labels)
checkpoint = torch.load("cnn_pnn_model.pth", map_location=device, weights_only=False)

# ✅ Load CNN in Feature Extraction Mode
num_classes = len(checkpoint.get('class_names', []))
cnn = create_cnn(num_classes, feature_extraction=True).to(device)
cnn.load_state_dict(checkpoint['cnn'])
cnn.eval()

# ✅ Load PNN Features and Labels
features = np.array(checkpoint['features'])
labels = np.array(checkpoint['labels'])
class_names = checkpoint['class_names']

# ✅ Initialize and train PNN
pnn = PNN()
pnn.fit(features, labels)

# ✅ Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # ✅ Extract CNN Features
    with torch.no_grad():
        feature = cnn(image).cpu().numpy().flatten()
    
    # ✅ Predict with PNN
    label_idx = pnn.predict(feature)
    predicted_class = class_names[label_idx]

    print(f"Predicted Class: {predicted_class}")
    return predicted_class

# ✅ Test with an image
image_path = r"D:\Final_Year_Project\Animal_footprint_detection\data\cropped_imgs\test\mink\184.jpg"
predicted_animal = predict_image(image_path)

print(f"Final Prediction: {predicted_animal}")
