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
from lib import summary_

def Get_opt():
    parse = argparse.ArgumentParser()
    parse.add_argument('--test_batch_size', type=int, default=64)
    parse.add_argument('--img_size', type=int, default=224)
    parse.add_argument('--model_name', type=str, default="mobilenet")
    parse.add_argument('--test_model', default="demo_model.pth")
    parse.add_argument('--data_root', default="./data")
    parse.add_argument('--test_list_file', default="./test.txt")
    parse.add_argument('--label_list_file', default="./label.txt")
    parse.add_argument('--GPU', type=bool, default=False)
    parse.add_argument('--score_th', type=float, default=0.0)
    return parse.parse_args()

def main():
    opt = Get_opt()
    task_root = "./"

    opt.model_name = "Conv2fc1_avg_16"
    opt.test_model = os.path.join(task_root, "out/car_Conv2fc1_avg_16_1230_40.pth")
    opt.data_root = os.path.join(task_root, "data/car/Data_hand")
    opt.test_list_file = os.path.join(task_root, "data/car/test.txt")
    opt.label_list_file = os.path.join(task_root, "data/car/label.txt")
    # opt.GPU = True
    opt.GPU = False
    opt.score_th = 0.4
    opt.img_size = 112

    test = test_.test_net(opt)
    m = nn.LogSoftmax()

    tp = 0
    fp = 0
    img_list = open(opt.test_list_file).readlines()
    num = len(img_list)

    # single test
    # first_time = time.time()
    # sum_a = 0
    # for i, line in enumerate(img_list):
    #     # start_time = time.time()
    #     img_name, index = line.split()
    #     img_path = os.path.join(opt.data_root, img_name)
    #     pred, score, a = test.single_test_(img_path)
    #     sum_a += a
    #     # print ("Single test time: {}".format(time.time()- start_time))
    #     # out_softmax = m(out)
    #     # print img_name, index
    #     # print pred, score
    #     if pred == int(index):
    #         tp +=1
    #     else:
    #         fp+=1
    #         # print i
    #     # print "======================="*2
    # print ("Single test time: {}".format((time.time()- first_time)/num))
    # print (1.0*tp/(tp+fp))
    # print (sum_a / (tp + fp))

    # batch test
    start_time = time.time()
    pred, score, true = test.batch_test_()
    print ("Single test time: {}".format((time.time() - start_time)/num))
    print (np.sum(np.abs(pred-true)))
    for i, label in enumerate(pred):
        if label != true[i]:
            print (i, true[i], label, score[i])

    np.save("./test_summary_0.4_pred.npy", pred)
    np.save("./test_summary_0.4_true.npy", true)
    np.save("./test_summary_0.4_score.npy", score)

def test_summary():
    pred = np.load("./test_summary_0.4_pred.npy")
    true = np.load("./test_summary_0.4_true.npy")
    score = np.load("./test_summary_0.4_score.npy")
    label = [1, 0]
    score_th = None
    summary = summary_.result_summary(pred, true, score,labels=label, score_th = score_th)

    print (len(pred))
    print (summary.Get_Confusion_matrix())
    # print (summary.diag())
    # print (summary.Get_labels())
    print ("TP:", summary.Get_TP())
    print ("FP:", summary.Get_FP())
    print ("FN:", summary.Get_FN())
    print ("ACC:", summary.Get_Accuracy())
    print ("Precision:", summary.Get_Precision())
    print ("Recall:", summary.Get_Recall())
    print ("AP:", summary.Get_AP())

if __name__ == '__main__':
    # main()
    test_summary()
