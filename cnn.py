import torch.nn as nn
import torchvision.models as models

def create_cnn(num_classes, feature_extraction=False):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if feature_extraction:
        model.fc = nn.Identity()
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
