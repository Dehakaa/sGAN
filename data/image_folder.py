"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data
from PIL import Image
import os

SUPPORTED_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF', '.mat'
]

def is_valid_image_file(filename):
    """Check if a file has a supported image extension."""
    return any(filename.lower().endswith(extension) for extension in SUPPORTED_EXTENSIONS)

def make_dataset(directory, max_size=float("inf")):
    """Recursively collect image file paths from a directory."""
    image_paths = []
    assert os.path.isdir(directory), f'{directory} is not a valid directory'

    for root, _, filenames in sorted(os.walk(directory)):
        for filename in filenames:
            if is_valid_image_file(filename):
                full_path = os.path.join(root, filename)
                image_paths.append(full_path)
    return image_paths[:min(max_size, len(image_paths))]

def default_image_loader(path):
    """Default image loader function."""
    return Image.open(path).convert('RGB')

class CustomImageDataset(data.Dataset):
    """A custom dataset class that loads images from a folder."""

    def __init__(self, root_dir, transform=None, return_paths=False, loader=default_image_loader):
        """Initialize the dataset with images from the given folder."""
        image_paths = make_dataset(root_dir)
        if len(image_paths) == 0:
            raise RuntimeError(f"Found 0 images in: {root_dir}\n"
                               f"Supported image extensions are: {', '.join(SUPPORTED_EXTENSIONS)}")

        self.root_dir = root_dir
        self.image_paths = image_paths
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, idx):
        """Return a transformed image and, optionally, its path."""
        path = self.image_paths[idx]
        image = self.loader(path)
        if self.transform:
            image = self.transform(image)
        if self.return_paths:
            return image, path
        return image

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_paths)

