# -*- coding: utf-8 -*-
import os
import torch
import _init_paths
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn
from net import my_net, my_tinynet
from lib import utils
from lib.data_load import myImageFloder
import time
from collections import OrderedDict
from PIL import Image
import sys
import numpy as np


class test_net():
    def __init__(self,opt):
        # 超参数(Hyperparameters)
        self.test_batch_size = opt.test_batch_size
        self.img_size = opt.img_size

        # 模型导入, 模型选择
        # self.model = my_net.MobileNet()
        # self.model = my_tinynet.Conv3fc2()
        self.model = getattr(sys.modules["net.my_tinynet"], opt.model_name)()
        self.model.load_state_dict(torch.load(opt.test_model, map_location=lambda storage, loc: storage))
        self.softmax_layer = nn.Softmax(dim=1)
        self.use_GPU = (torch.cuda.is_available() and opt.GPU)

        if self.use_GPU:
            self.model = self.model.cuda()
        self.model.eval()

        self.data_root = opt.data_root
        self.test_list_file = opt.test_list_file

        label_list = open(opt.label_list_file, "r").readlines()
        self.label_index = OrderedDict()
        self.index_label = OrderedDict()
        for line in label_list:
            label, index = line.split()
            index = int(index)
            self.label_index[label] = index
            self.index_label[index] = label
        self.class_num = len(label_list)
        self.label_th = True
        self.score_th = opt.score_th

    def batch_data_load(self):
        # 数据预处理 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        img_transform = transforms.Compose([
            transforms.Resize(int(self.img_size*1.1)),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        data_folder = myImageFloder(self.data_root, self.test_list_file, transform=img_transform)

        self.dataloader = DataLoader(data_folder, batch_size=self.test_batch_size, shuffle=False, num_workers=4)
        self.test_num = len(data_folder)

    def single_data_load(self,img_path):
        # 数据预处理
        img_transform = transforms.Compose([
            transforms.Resize(int(self.img_size*1.1)),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])

        self.img = Image.open(img_path).convert('RGB')
        self.img = img_transform(self.img).unsqueeze(0)

    def Get_pred_label(self,out):
        out_ = out.clone()
        # out_[out_<self.score_th] = 0.0
        _, pred = torch.max(out_, 1)
        index = _ < self.score_th
        if sys.version_info > (3, ):
            pred[index] = 0
        else:
            if torch.sum(index) != 0:
                pred[index] = 0
        return out[[range(len(pred)),pred]], pred

    def batch_test_(self):
        self.batch_data_load()
        # pred_label = Variable(torch.LongTensor(0, 1))
        # pred_score = Variable(torch.FloatTensor(0, 1))
        # true_label = Variable(torch.LongTensor(0, 1))
        # if self.use_GPU:
        #     pred_label = pred_label.cuda()
        #     true_label = true_label.cuda()
        for i, data in enumerate(self.dataloader):
            img, label = data
            if self.use_GPU:
                img = Variable(img, volatile=True).cuda()
                label = Variable(label, volatile=True).cuda()
            else:
                img = Variable(img, volatile=True)
                label = Variable(label, volatile=True)
            out = self.model(img)
            out = self.softmax_layer(out)
            # score, pred = torch.max(out, 1)
            score, pred = self.Get_pred_label(out)
            if i == 0:
                pred_score = score
                pred_label = pred
                true_label = label
            else:
                pred_score = torch.cat((pred_score, score),0)
                pred_label = torch.cat((pred_label, pred),0)
                true_label = torch.cat((true_label, label), 0)
        if self.use_GPU:
            pred_score = pred_score.cpu()
            pred_label = pred_label.cpu()
            true_label = true_label.cpu()
        return pred_label.data.numpy(), pred_score.data.numpy(), true_label.data.numpy()

    def single_test_(self, img_path):
        load_time_start = time.time()
        self.single_data_load(img_path=img_path)
        load_time = time.time() - load_time_start
        if self.use_GPU:
            img = Variable(self.img, volatile=True).cuda()
        else:
            img = Variable(self.img, volatile=True)
        out = self.model(img)
        out = self.softmax_layer(out)
        score, pred = self.Get_pred_label(out)
        if self.use_GPU:
            pred = pred.cpu()
            score = score.cpu()
        return pred.data.numpy(), score.data.numpy(), load_time

    def summary_params(self):
        params = self.model.state_dict()
        sum_p = 0
        for k,v in params.items():
            p_i = 1
            for i in range(v.dim()):
                p_i *= v.size()[i]
            sum_p += p_i
        return sum_p
