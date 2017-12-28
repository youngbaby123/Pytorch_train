import os
import h5py
import numpy as np
import argparse

import torch
from torchvision import models, transforms
from torch import optim, nn
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from lib.data_load import myImageFloder
from torch.utils.data import DataLoader

parse = argparse.ArgumentParser()
# parse.add_argument(
#     '--model', required=True, help='vgg, inceptionv3, resnet152')
parse.add_argument('--bs', type=int, default=5)
# parse.add_argument('--phase', required=True, help='train, val')
opt = parse.parse_args()
print(opt)

img_transform = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

root = '/home/zkyang/Workspace/task/Pytorch_task/Pytorch_train/data/car/Data'
train_txt = '/home/zkyang/Workspace/task/Pytorch_task/Pytorch_train/data/car/train.txt'
val_txt = '/home/zkyang/Workspace/task/Pytorch_task/Pytorch_train/data/car/val.txt'
data_folder = {
    'train': myImageFloder(root, train_txt, transform=img_transform),
    'val': myImageFloder(root, val_txt, transform=img_transform)
}

# root = '/home/zkyang/Workspace/task/Pytorch_task/test_m/data/car'
# data_folder = {
#     'train': ImageFolder(os.path.join(root, 'train'), transform=img_transform),
#     'val': ImageFolder(os.path.join(root, 'val'), transform=img_transform)
# }

# define dataloader to load images
batch_size = opt.bs
dataloader = {
    'train':
    DataLoader(
        data_folder['train'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=4),
    'val':
    DataLoader(
        data_folder['val'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)
}

# get train data size and validation data size
data_size = {
    'train': len(dataloader['train'].dataset),
    'val': len(dataloader['val'].dataset)
}

# get numbers of classes
img_classes = len(dataloader['train'].dataset.classes)
print dataloader['train'].dataset.classes

# test if using GPU
use_gpu = torch.cuda.is_available()

# for i, data in enumerate(dataloader['train'], 1):
#     img, label = data
#     print img
print dataloader['train'].dataset.classes
print (len(data_folder["train"]))
print (len(data_folder["val"]))