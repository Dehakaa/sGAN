

# sGAN
source code of Enhanced Flow Visualization Using Image Processing and Deep Learning Techniques
这个是我用来保存和分享上述论文中所需要的sGAN代码部分的内容，至于XX部分，请访问XXX

为了让代码成功运行起来，你可能需要如下环境：

GAN是一种非常强大的神经网络

《Intro of sGAN》
一些效果图

**Note**: The current software works well with PyTorch-cuda 12.1. It may have some trouble if using older version.

## Prerequisites
- Linux or Windows
- Python 3.8
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/Dehakaa/sGAN
cd sGAN
```

- Install [PyTorch](http://pytorch.org) and 0.4+ and other.
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.
  

### to train/test sGAN
- Download a sGAN dataset (you may find temp files follow the [guidance](https://github.com/Dehakaa/synimage) ):
```bash
bash ./datasets/download_sgan_dataset.sh temp
```
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.
- To log training progress and test images to W&B dashboard, set the `--use_wandb` flag with train and test script
- Train a temp model:
```bash
#!./scripts/train_sgan.sh
python train.py --dataroot ./datasets/temps --name temps_train --model s_gan
```
If you want to get more information about the training process, please refer to [Wandb](https://wandb.com).
- Test the model:
```bash
#!./scripts/test_sgan.sh
python test.py --dataroot ./datasets/temps --name temps_train --model s_gan
```

- Test the results
```bash
python test.py --dataroot datasets/temps/testA --name cascade_pretrained --model test --no_dropout
```


## Acknowledgments
Our code is inspired by [pytorch-CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
