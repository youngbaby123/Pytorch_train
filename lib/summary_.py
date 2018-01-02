# -*- coding: utf-8 -*-
import os
import torch
import _init_paths
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn
from net import my_net, my_tinynet
import utils
from data_load import myImageFloder
import time
from collections import OrderedDict
from PIL import Image
from sklearn.metrics import confusion_matrix

class result_summary():
    def __init__(self,pred,true):
        self.pred = pred
        self.true = true
        # self.result_dict = {}
        # for index_i, true_i in enumerate(true):
        #     if not true_i in self.result_dict:
        #         self.result_dict[true_i]["tp"] = 0
        #         self.result_dict[true_i]["fp"] = 0
        #         self.result_dict[true_i]["total"] = 0
        #     if self.pred[index_i] == true_i:
        #         self.result_dict[true_i] += 1



    def Get_TP(self):
        tp = 0
        for i, pred_i in  enumerate(self.pred):
            if pred_i == self.true[i]:
                tp += 1
        return tp

    # def Get_FP(self):
    def Get_Confusion_matrix(self):
        return confusion_matrix(self.true, self.pred,labels=[0,1])
