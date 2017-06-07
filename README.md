# U-Net implementation in PyTorch

This is a U-Net implementation as described in [Ronneberger et al.](https://arxiv.org/pdf/1505.04597.pdf) using PyTorch. While the code was used to train on density images, it can be used for image segmentation by replacing the loss function.

The network is defined in `unet.py` and training is done in `train.py`.
