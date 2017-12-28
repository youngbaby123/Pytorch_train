# -*- coding: utf-8 -*-
import torch
import _init_paths
from torch import nn, optim
from net import my_net


def model(model_name):
    if model_name in ["mobilenet", 0]:
        return my_net.MobileNet()

# 各优化方式的函数定义
def opt_algorithm(finetune_params, learning_rate, alg_name):
    if alg_name in ["SGD", 0]:
        return optim.SGD(params=finetune_params, lr=learning_rate, momentum=0.9)
    elif alg_name in ["ADAM", 1]:
        return optim.Adam(params=finetune_params, lr=learning_rate, betas=(0.9, 0.99))
    elif alg_name in ["RMSprop", 2]:
        return optim.RMSprop(params=finetune_params, lr=learning_rate, alpha=0.9)

def loss_function(loss_name):
    label_th = True
    if loss_name in ["CrossEntropyLoss", 0]:
        label_th = False
        return nn.CrossEntropyLoss(), label_th
    elif loss_name in ["L1Loss", 1]:
        return nn.L1Loss(), label_th
    elif loss_name in ["SmoothL1Loss", 2]:
        return nn.SmoothL1Loss(), label_th
    elif loss_name in ["MSELoss", 3]:
        return nn.MSELoss(), label_th


def label_to_vector(label, label_num):
    res = torch.zeros(label.numel(), label_num)
    for i,label_i in enumerate(label):
        res[i][label_i] = 1.0
    return res