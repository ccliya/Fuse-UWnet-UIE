
import os
from PIL import Image
import torchvision.transforms as transforms
import torch

def preprocess_image(image_path, image_size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)  # [1, 3, H, W]
