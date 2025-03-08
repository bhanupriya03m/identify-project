import torch
import torch.nn as nn
import torchvision.models as models

def create_cnn(num_classes):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Replace final layer
    return model
