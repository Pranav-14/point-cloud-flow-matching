import os

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

from dit import DiT


def load_images(n, image_size=32, root="~/fastdata/cifar10"):
    """First n CIFAR-10 training images as one tensor (n, 3, H, W) in [-1, 1]."""
    tf = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    ds = datasets.CIFAR10(root=os.path.expanduser(root), train=True, download=True, transform=tf)
    imgs = torch.stack([ds[i][0] for i in range(n)])
    return imgs * 2 - 1


def build_model(args):
    return DiT(
        image_size=args.image_size,
        patch_size=args.patch_size,
        hidden=args.hidden,
        depth=args.depth,
        heads=args.heads,
    )


def save_grid(imgs, path, nrow=8):
    save_image((imgs.clamp(-1, 1) + 1) / 2, path, nrow=nrow)


def nearest_train_mse(samples, train):
    """For each sample, the MSE to its closest training image.
    When overfitting, a good model gets this close to 0 (it reproduces the training set)."""
    d = torch.cdist(samples.flatten(1), train.flatten(1)) ** 2 / samples[0].numel()  # (S, N) per-pixel MSE
    mse, idx = d.min(dim=1)
    return mse, idx
