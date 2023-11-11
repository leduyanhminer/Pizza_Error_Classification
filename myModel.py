import torch
import torch.nn as nn
import torchvision


def create_model():
    model = torchvision.models.resnet50()
    num_classes = 20
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load('model/best_model_ver2.pth'))

    return model


