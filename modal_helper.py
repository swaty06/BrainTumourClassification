

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

trained_model = None
class_names =['No Tumour','Tumour Present']


# Load the pre-trained ResNet model
def predict(image_path, device='cpu'):
    global trained_model

    # Define the same model architecture used during training
    if trained_model is None:
        trained_model = models.resnet18(pretrained=False)
        trained_model.fc = nn.Linear(trained_model.fc.in_features, 2)  # Binary classification
        trained_model.load_state_dict(torch.load("saved_modelbrain.pth", map_location=device))
        trained_model.to(device)
        trained_model.eval()

    # Preprocess the image
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = trained_model(image_tensor)
        _, predicted = torch.max(output, 1)
        return class_names[predicted.item()]