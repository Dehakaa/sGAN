import os
from data.temp_data import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import scipy.io
import numpy as np
import torch
load_size = 1080
crop_size = 1080
from torchvision.transforms import v2

transforms = v2.Compose([
    v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
    v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
    # ...
    v2.RandomResizedCrop(size=(224, 224), antialias=True),  # Or Resize(antialias=True)
    # ...
    v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


A_path = 'D:\\project\\data\\test_trainset\\train\\Img\\1.jpg'
B_path = 'D:\\project\\data\\test_trainset\\train\\Img\\2.jpg'
A = Image.open(A_path).convert('RGB')
B = Image.open(B_path).convert('RGB')

A_ = transforms(A)
B_ = transforms(B)