# -*- coding: utf-8 -*-
import os
import _init_paths
from lib import train_
import logging
import argparse
from collections import OrderedDict
import pickle
from net import my_tinynet

def Get_opt():
    parse = argparse.ArgumentParser()
    parse.add_argument('--model_name', type=str, default="mobilenet")
    parse.add_argument('--train_batch_size', type=int, default=64)
    parse.add_argument('--val_batch_size', type=int, default=64)
    parse.add_argument('--img_size', type=int, default=224)
    parse.add_argument('--learning_rate', type=float, default=1e-2)
    parse.add_argument('--num_epoches', type=int, default=20)
    parse.add_argument('--step_size', type=int, default=5)
    parse.add_argument('--learning_rate_dec', type=float, default=0.8)
    # ["CrossEntropyLoss", "L1Loss", "SmoothL1Loss", "MSELoss"]
    parse.add_argument('--loss_name', default="CrossEntropyLoss")
    parse.add_argument('--finetune', type=bool, default=False)
    parse.add_argument('--finetune_model', type=str, default="demo_model")
    parse.add_argument('--finetune_layer_num', type=int, default=-1)
    # [SGD, ADAM, RMSprop]
    parse.add_argument('--alg_name', default="SGD")
    parse.add_argument('--data_root', default="./data")
    parse.add_argument('--train_list_file', default="./train.txt")
    parse.add_argument('--val_list_file', default="./val.txt")
    parse.add_argument('--label_list_file', default="./label.txt")
    parse.add_argument('--save_model_step',type=int, default=5)
    parse.add_argument('--save_model_path', default="./out")
    parse.add_argument('--save_model_name', default="demo_model")

    parse.add_argument('--save_train_loss', type=bool, default=False)
    parse.add_argument('--save_train_path', type=str, default="./result")
    return parse.parse_args()


def for_start():
    opt = Get_opt()
    net_list = open("./net/netlistbak_2.txt", "r").readlines()
    for list_i in net_list:
        netname, img_size = list_i.split()
        if_trainbatch = netname.split("_")[-1]
        opt.model_name = netname
        opt.img_size = int(img_size)
        opt.save_model_name = "car_{}_1230".format(netname)
        print (netname, opt.finetune)
        if netname in ["MobileNet_dw8", "MobileNet_dw5", "MobileNet_dw3"]:
            opt.finetune = True
            opt.learning_rate = 1e-2
        else:
            opt.finetune = False
            opt.learning_rate = 1e-2
        if if_trainbatch =="64":
            opt.train_batch_size = 32
        else:
            opt.train_batch_size = 64
        print (opt.finetune)
        Start_train_(opt)



def Start_train_(opt):
    # opt = Get_opt()
    res = OrderedDict()
    train_class = train_.train_net(opt)
    print ("==" * 36)
    print(" alg_name:  {}\n train_batch_size:  {}\n num_epoches:  {}\n step_size:  {}\n learning_rate_dec:  {}".format(
        opt.alg_name, opt.train_batch_size, opt.num_epoches, opt.step_size, opt.learning_rate_dec
    ))
    print (opt)
    # try:
    print ("Start train net!")
    train_loss, train_acc, val_loss, val_acc = train_class.train_all()
    print ("==" * 36)
    print ("Done!")
    name = "{}_{}_{}_{}_{}".format(opt.alg_name, opt.train_batch_size, opt.num_epoches, opt.step_size, opt.learning_rate_dec)
    res[name] = {}
    res[name]["train_loss"] = train_loss
    res[name]["train_acc"] = train_acc
    res[name]["val_loss"] = val_loss
    res[name]["val_acc"] = val_acc
    print ("==" * 36)
    print ("train loss: \n{}".format(train_loss))
    print ("------------------------------------")
    print ("train acc: \n{}".format(train_acc))
    print ("------------------------------------")
    print ("val loss: \n{}".format(val_loss))
    print ("------------------------------------")
    print ("val acc: \n{}".format(val_acc))
    # except Exception as error:
    #     logging.error('train error.')
    #     return
    if opt.save_train_loss:
        res_save_path = opt.save_train_path
        if not os.path.exists(res_save_path):
            os.mkdir(res_save_path)
        f1 = open('{}/{}_{}.pkl'.format(res_save_path, opt.save_model_name, name), 'wb')
        pickle.dump(res, f1)


def main():
    # opt = Get_opt()
    # test_demo(opt)
    Start_train_()


if __name__ == '__main__':
    # main()
    for_start()
