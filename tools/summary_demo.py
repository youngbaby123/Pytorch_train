# -*- coding: utf-8 -*-
import os
import _init_paths
from lib import test_
import logging
import argparse
from collections import OrderedDict
import torch
from torch import nn
import numpy as np
import time
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
        for summary_label_i in ["accuracy", "test_speed", "load_speed", "params_num", "file_size"]:
            save_res_[i].append(str(res[summary_label_i]))
    save_txt = []
    for i in save_res_:
        save_i = "\t".join(save_res_[i])
        save_txt.append(save_i)
    file.write("\n".join(save_txt)+"\n")
    file.close()


def get_FileSize(filePath):
    # filePath = unicode(filePath,'utf8')
    fsize = os.path.getsize(filePath)
    fsize = fsize/float(1024)

    return round(fsize,2)

def all_summary():
    opt = Get_opt()
    this_dir = os.path.dirname(__file__)
    task_root = os.path.join(this_dir, "..")
    opt.data_root = os.path.join(task_root, "data/train_demo/Data")
    opt.test_list_file = os.path.join(task_root, "data/train_demo/test.txt")
    opt.label_list_file = os.path.join(task_root, "data/train_demo/label.txt")
    # opt.GPU = True
    opt.GPU = False

    img_list = open(opt.test_list_file).readlines()

    model_list_file = open(os.path.join(task_root, "net/netlist_tiny.txt"), "r").readlines()

    summary_label = ["model_name", "label", "tp", "fp", "fn", "precision", "recall", "AP", "accuracy", "test_speed",
                     "load_speed", "params_num", "file_size"]
    save_file_path = os.path.join(task_root, "data/train_demo/summary_CPU.txt")
    save_file = open(save_file_path, "w+")
    save_file.write("\t".join(summary_label)+"\n")
    save_file.close()

    res = OrderedDict()

    for model_i in model_list_file:
        opt.model_name, img_size = model_i.split()
        opt.test_model = os.path.join(task_root, "out/car_{}_1230_40.pth".format(opt.model_name))
        opt.img_size = int(img_size)
        # opt.score_th = 0.8
        # 是否启动GPU
        opt.GPU = False
        test = test_.test_net(opt)
        res[opt.model_name] = {}

        #模型的参数个数统计以及模型存储大小
        res[opt.model_name]["params_num"] = test.summary_params()
        res[opt.model_name]["file_size"] = get_FileSize(opt.test_model)
        # time test 时间测试, 从第101张到第1100张的平均测试时间,平均数据导入时间
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
        res[opt.model_name]["test_speed"] = (end_time - start_time)
        res[opt.model_name]["load_speed"] = (sum_load_time)
        print ("Average single test time: {}".format(res[opt.model_name]["test_speed"]))
        print ("Average single data load time: {}".format(res[opt.model_name]["load_speed"]))

        # batch test
        opt.GPU = True
        test = test_.test_net(opt)
        pred, score, true = test.batch_test_()
        score_th = None
        label = [1, 0]
        summary = summary_.result_summary(pred, true, score, labels=label, score_th=score_th)
        res[opt.model_name]["tp"] = summary.Get_TP()
        res[opt.model_name]["fp"] = summary.Get_FP()
        res[opt.model_name]["fn"] = summary.Get_FN()
        res[opt.model_name]["precision"] = summary.Get_Precision()
        res[opt.model_name]["recall"] = summary.Get_Recall()
        res[opt.model_name]["accuracy"] = summary.Get_Accuracy()
        res[opt.model_name]["AP"] = summary.Get_AP()
        print (res[opt.model_name])

        save_res(res[opt.model_name], opt.model_name, save_file_path)


if __name__ == '__main__':
    all_summary()
