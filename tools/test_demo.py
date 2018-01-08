# -*- coding: utf-8 -*-
import os
from lib import test_
import logging
import argparse
from collections import OrderedDict
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


def single_test():
    opt = Get_opt()
    this_dir = os.path.dirname(__file__)
    task_root = os.path.join(this_dir, "..")
    # 检测图像根目录
    opt.data_root = os.path.join(task_root, "data/car/Data_more")
    # 检测图像列表
    opt.test_list_file = os.path.join(task_root, "data/car/sm_list.txt")
    img_list = open(opt.test_list_file).readlines()
    # 类别及其对应编号列表
    opt.label_list_file = os.path.join(task_root, "data/car/label.txt")
    # GPU切换
    # opt.GPU = True
    opt.GPU = False

    opt.model_name = "DwNet112_dw3_avg_16"
    opt.test_model = os.path.join(task_root, "out/out_40/car_{}_1230_40.pth".format(opt.model_name))
    opt.img_size = 112

    # 导入模型，并初始化
    test = test_.test_net(opt)

    # 统计模型参数个数以及模型存储大小
    params_num = test.summary_params()
    file_size = get_FileSize(opt.test_model)
    # 模型参数个数
    print ("Model params num: {}".format(params_num))
    # 模型存储大小
    print ("Model file size: {}KB".format(file_size))

    # 单张图片检测
    for i, line in enumerate(img_list):
        start_time = time.time()
        img_name, index = line.split()
        img_path = os.path.join(opt.data_root, img_name)
        # 单张图片进行检测
        pred, score, load_time = test.single_test_(img_path)
        test_time = time.time() - start_time
        # 检测结果
        print ("图像路径： {}".format(img_path))
        print ("检测结果汇总\n \t所属编号：{}\n\t所属类别：{}\n\t所属分数： {:.3f}".format(pred[0], test.index_label[pred[0]], score[0]))
        # 运行时间统计
        print (" 总运行时间： {:.3f}ms\n 图片导入及预处理时间： {:.3f}ms\n 模型运算时间： {:.3f}ms".format(test_time*1000, load_time*1000, (test_time - load_time)*1000))
        print ("----------------"*4)


def get_FileSize(filePath):
    # filePath = unicode(filePath,'utf8')
    fsize = os.path.getsize(filePath)
    fsize = fsize/float(1024)

    return round(fsize,2)

if __name__ == '__main__':
    single_test()
