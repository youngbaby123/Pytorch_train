# -*- coding: utf-8 -*-
import os
from PIL import Image
import cv2
import random
import argparse
from collections import OrderedDict

def parse_args():
    """
    parse input arguments
    """
    parser = argparse.ArgumentParser(description='Pre-process the classfication dataset')
    parser.add_argument('--data_root', dest='data_root',
                        default=None, type=str,
                        help='Data root to be preprocessed')
    parser.add_argument('--train_ratio', dest='train_ratio',
                        default=0.8, type=float,
                        help='Ratio of the training data')
    parser.add_argument('--val_ratio', dest='val_ratio',
                        default=0.1, type=float,
                        help='Ratio of the validation data')
    parser.add_argument('--test_ratio', dest='test_ratio',
                        default=0.1, type=float,
                        help='Ratio of the test data')
    parser.add_argument('--save_root', dest='save_root',
                        default='./', type=str,
                        help='the train/val/test list path to be saved')

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args


def check_img(img_path):
    try:
        img = Image.open(img_path)
        img.verify()
        img = cv2.imread(img_path)
        if not img is None:
            return True
        else:
            return False
    except Exception, e:
        print e.message
        return False

def load_file(root, rel_path = "", img_list=[], rel_img_list = []):
    if os.path.isfile(root):
        if check_img(root):
            img_list.append(root)
            rel_img_list.append(rel_path)
    elif os.path.isdir(root):
        for path_i in os.listdir(root):
            sub_root = os.path.join(root, path_i)
            sub_rel_path = os.path.join(rel_path, path_i)
            img_list, rel_img_list = load_file(sub_root, sub_rel_path, img_list, rel_img_list)
    return img_list, rel_img_list

def save_list(all_data_dict, args):
    label_list = []
    train_list = []
    val_list = []
    test_list = []
    for label_ in all_data_dict:
        label_index = all_data_dict[label_]["label_index"]
        sub_file_list = all_data_dict[label_]["file_list"]
        random.shuffle(sub_file_list)
        num_ = len(sub_file_list)
        last_train_index = int(num_ * args.train_ratio)
        last_val_index = int(num_ * (args.train_ratio + args.val_ratio))
        sub_train_list = ["{} {}".format(file_i, label_index) for file_i in sub_file_list[: last_train_index]]
        sub_val_list = ["{} {}".format(file_i, label_index) for file_i in sub_file_list[last_train_index : last_val_index]]
        sub_test_list = ["{} {}".format(file_i, label_index) for file_i in sub_file_list[last_val_index :]]
        # print sub_val_list
        if label_index == 0:
            label_list.insert(0, "{} {}".format(label_, 0))
            train_list = sub_train_list + train_list
            val_list = sub_val_list + val_list
            test_list = sub_test_list + test_list
        else:
            label_list.append("{} {}".format(label_, label_index))
            train_list += sub_train_list
            val_list += sub_val_list
            test_list += sub_test_list
    open(os.path.join(args.save_root,"train.txt"),"wb+").write("\n".join(train_list))
    open(os.path.join(args.save_root, "val.txt"), "wb+").write("\n".join(val_list))
    open(os.path.join(args.save_root, "test.txt"), "wb+").write("\n".join(test_list))
    open(os.path.join(args.save_root, "label.txt"), "wb+").write("\n".join(label_list))


def Get_data_list(args):
    label_name = [label_i for label_i in os.listdir(args.data_root)]
    list.sort(label_name)
    index_ = 1

    data_set = OrderedDict()
    for label_ in label_name:
        if label_ in ["0", "background", "fake_background", "Negative", "negative"]:
            label_index = 0
        else:
            label_index = index_
            index_ += 1
        sub_data_root = os.path.join(args.data_root, label_)
        data_set[label_] = {}
        _, data_set[label_]["file_list"] = load_file(sub_data_root, rel_path = label_, img_list=[], rel_img_list = [])
        data_set[label_]["label_index"] = label_index
    return data_set


def main():
    args = parse_args()
    # args.data_root = "/home/zkyang/Workspace/task/Pytorch_task/test_m/data/car"
    # args.save_root = "/home/zkyang/Workspace/task/Pytorch_task/test_m/data"

    print ("Start get data list.")
    data_set = Get_data_list(args)
    print ("Get done.")
    print ("Start save data list.")
    save_list(data_set, args)
    print ("Save done.")


if __name__ == '__main__':
    main()
