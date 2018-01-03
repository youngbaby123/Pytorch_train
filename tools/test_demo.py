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


    img_list = open(opt.test_list_file).readlines()
    num = len(img_list)

    # # single test
    # tp = 0
    # fp = 0
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



def save_res(res, model_name, save_path):
    file = open(save_path, "a+")
    save_res_ = OrderedDict()
    for i in res["tp"]:
        save_res_[i]=[]
        save_res_[i].append(str(model_name))
        save_res_[i].append(str(i))
    for i in save_res_:
        for summary_label_i in ["tp", "fp", "fn", "precision", "recall", "AP"]:
            save_res_[i].append(str(res[summary_label_i][i]))
        for summary_label_i in ["accuracy", "test_speed", "load_speed"]:
            save_res_[i].append(str(res[summary_label_i]))
    save_txt = []
    for i in save_res_:
        save_i = "\t".join(save_res_[i])
        save_txt.append(save_i)
    file.write("\n".join(save_txt)+"\n")
    file.close()


def all_summary():
    opt = Get_opt()
    task_root = "/home/zkyang/Workspace/task/Pytorch_task/Pytorch_train"
    opt.data_root = os.path.join(task_root, "data/car/Data_hand")
    opt.test_list_file = os.path.join(task_root, "data/car/test.txt")
    opt.label_list_file = os.path.join(task_root, "data/car/label.txt")
    # opt.GPU = True
    opt.GPU = False

    img_list = open(opt.test_list_file).readlines()

    model_list_file = open(os.path.join(task_root, "net/netlist.txt"), "r").readlines()

    summary_label = ["model_name", "label", "tp", "fp", "fn", "precision", "recall", "AP", "accuracy", "test_speed",
                     "load_speed"]
    save_file_path = os.path.join(task_root, "all_summary.txt")
    save_file = open(save_file_path, "w+")
    save_file.write("\t".join(summary_label)+"\n")
    save_file.close()

    res = OrderedDict()

    for model_i in model_list_file:
        opt.model_name, img_size = model_i.split()
        opt.test_model = os.path.join(task_root, "out/out_40/car_{}_1230_40.pth".format(opt.model_name))
        opt.img_size = int(img_size)
        opt.score_th = 0.8
        test = test_.test_net(opt)
        res[opt.model_name] = {}

        # time test 时间测试, 从第11张到第110张的平均测试时间,平均数据导入时间
        sum_load_time = 0
        for i, line in enumerate(img_list[:1100]):
            if i == 100:
                start_time = time.time()

            img_name, index = line.split()
            img_path = os.path.join(opt.data_root, img_name)
            pred, score, load_time = test.single_test_(img_path)
            if i > 99:
                sum_load_time += load_time

        end_time = time.time()
        res[opt.model_name]["test_speed"] = (end_time - start_time) / 100
        res[opt.model_name]["load_speed"] = (sum_load_time) / 100
        print ("Average single test time: {}".format(res[opt.model_name]["test_speed"]))
        print ("Average single data load time: {}".format(res[opt.model_name]["load_speed"]))

        # batch test
        pred, score, true = test.batch_test_()
        score_th = None
        label = [1,0]
        summary = summary_.result_summary(pred, true, score, labels=label, score_th=score_th)
        res[opt.model_name]["tp"] = summary.Get_TP()
        res[opt.model_name]["fp"] = summary.Get_FP()
        res[opt.model_name]["fn"] = summary.Get_FN()
        res[opt.model_name]["precision"] = summary.Get_Precision()
        res[opt.model_name]["recall"] = summary.Get_Recall()
        res[opt.model_name]["accuracy"] = summary.Get_Accuracy()
        res[opt.model_name]["AP"] = summary.Get_AP()
        print res[opt.model_name]

        save_res(res[opt.model_name], opt.model_name, save_file_path)

if __name__ == '__main__':
    # main()
    # test_summary()
    all_summary()
