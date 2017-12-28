# -*- coding: utf-8 -*-
import os
from lib import test_
import logging
import argparse
from collections import OrderedDict
import torch
from torch import nn
import numpy as np
import time
import _init_paths

def Get_opt():
    parse = argparse.ArgumentParser()
    parse.add_argument('--test_batch_size', type=int, default=64)
    parse.add_argument('--img_size', type=int, default=224)
    parse.add_argument('--test_model', default="demo_model.pth")
    parse.add_argument('--data_root', default="./data")
    parse.add_argument('--test_list_file', default="./test.txt")
    parse.add_argument('--label_list_file', default="./label.txt")
    parse.add_argument('--GPU', type=bool, default=False)
    parse.add_argument('--score_th', type=float, default=0.8)
    return parse.parse_args()

def main():
    opt = Get_opt()
    task_root = "/home/zkyang/Workspace/task/Pytorch_task/Pytorch_train/data/test_demo"

    opt.test_model = os.path.join(task_root, "mobilenet_car_ft_fc_1227_7.pth")
    opt.data_root = "/home/zkyang/Workspace/task/Pytorch_task/Pytorch_train/data/car/Data"
    opt.test_list_file = os.path.join(task_root, "test.txt")
    opt.label_list_file = os.path.join(task_root, "label.txt")
    opt.GPU = True
#    opt.GPU = False
    opt.score_th = 0.8

    test = test_.test_net(opt)
    m = nn.LogSoftmax(dim=1)

    tp = 0
    fp = 0
    img_list = open(opt.test_list_file).readlines()
    num = len(img_list)

    # single test
    first_time = time.time()
    for i, line in enumerate(img_list):
        # start_time = time.time()
        img_name, index = line.split()
        img_path = os.path.join(opt.data_root, img_name)
        pred, score = test.single_test_(img_path)
        # print ("Single test time: {}".format(time.time()- start_time))
        # out_softmax = m(out)
        # print img_name, index
        print pred, score
        if pred == int(index):
            tp +=1
        else:
            fp+=1
            print i
        # print "======================="*2
    print ("Single test time: {}".format((time.time()- first_time)/num))
    print 1.0*tp/(tp+fp)

    # # batch test
    # start_time = time.time()
    # pred, true = test.batch_test_()
    # print ("Single test time: {}".format((time.time() - start_time)/num))
    # # print np.sum(pred-true)
    # for i, label in enumerate(pred):
    #     if label != true[i]:
    #         print i

if __name__ == '__main__':
    main()
