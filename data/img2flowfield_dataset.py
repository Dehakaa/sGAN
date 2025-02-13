import os
from data.base_dataset import BaseDataset, compute_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import scipy.io
import numpy as np


class Img2FlowfieldDataset(BaseDataset):
    """A dataset class for paired image and flowfield dataset.

    It assumes that the directory "/path/to/data/train/Img" contains images and
    the directory "/path/to/data/train/Flowfield" contains flow field data.
    During test time, you need to prepare a directory '/path/to/data/test/Img' and '/path/to/data/test/Flowfield'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.dir_A = os.path.join(self.dir_AB, 'Img')
        self.dir_B = os.path.join(self.dir_AB, 'Flowfield')
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # get image paths
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # get mat file paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        A_path = self.A_paths[index]
        A = Image.open(A_path).convert('RGB')
        B_path = self.B_paths[index]
        mat_data = scipy.io.loadmat(B_path)
        B = mat_data['combinedData']
        B = np.array(B, dtype=np.float64)
        # split AB image into A and B
        w, h = A.size
        # w2 = int(w / 2)
        # A = AB.crop((0, 0, w2, h))
        # B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = compute_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1), mat_file=True)

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
