import os
from data.temp_data import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from torchvision.transforms import v2
from PIL import Image
import scipy.io
import numpy as np
import torch
load_size = 1080
crop_size = 1080



A_path = 'D:\\project\\file_exchange3\\IMG\\0022.png'
B_path = 'D:\\project\\file_exchange3\\UV\\UV_0001.mat'
C_path = 'D:\\project\\data\\movies\\DSC_0066.JPG'
A = Image.open(A_path).convert('RGB')
C = Image.open(C_path).convert('RGB')
mat_data = scipy.io.loadmat(B_path)
U = mat_data['U']
V = mat_data['V']
B = np.stack((U, V))
B = torch.from_numpy(B)
A_transform = v2.Compose([
    v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
    v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
    # ...
    # v2.RandomResizedCrop(size=(224, 224), antialias=True),  # Or Resize(antialias=True)
    # ...
    v2.Resize(size=(200,200),antialias=True),
    v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

B_transform = v2.Compose([
    # v2.RandomResizedCrop(size=(224, 224), antialias=True),  # Or Resize(antialias=True)
    v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    v2.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]),
])

A_ = A_transform(A)
B_ = B_transform(B)
C_ = A_transform(C)


# Image Classification
# import torch
# from torchvision.transforms import v2
#
# H, W = 32, 32
# img = torch.rand(size=(2, H, W), dtype=torch.float32)
#
# trans = v2.Compose([
#     v2.RandomResizedCrop(size=(224, 224), antialias=True),
#     v2.RandomHorizontalFlip(p=0.5),
#     v2.ToDtype(torch.float32, scale=True),
#     v2.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]),
# ])
# img = trans(img)