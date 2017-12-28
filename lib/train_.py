# -*- coding: utf-8 -*-
import os
import torch
import _init_paths
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import utils
from data_load import myImageFloder
import time

class train_net():
    def __init__(self,opt):
        # 超参数(Hyperparameters)
        self.train_batch_size = opt.train_batch_size
        self.val_batch_size = opt.val_batch_size
        self.img_size = opt.img_size

        self.learning_rate = opt.learning_rate
        self.num_epoches = opt.num_epoches
        self.step_size = opt.step_size
        self.learning_rate_dec = opt.learning_rate_dec

        # 模型导入, 模型选择
        # self.model = my_net.MobileNet()
        self.model = utils.model(opt.model_name)

        self.finetune_params = [k for k in list(self.model.parameters())]
        print ("IF fine tune: {}!".format(opt.finetune))
        if opt.finetune:
            self.model.load_state_dict(torch.load(opt.finetune_model))
            fine_num = -opt.finetune_layer_num
            if opt.finetune_layer_num == -1:
                fine_num = 0
            for para in list(self.model.parameters())[:fine_num]:
                para.requires_grad = False
                self.finetune_params.pop(0)
            # print self.finetune_params

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.loss_name = opt.loss_name
        self.criterion, self.label_th = utils.loss_function(self.loss_name)

        # 优化算法选择
        self.alg_name = opt.alg_name

        self.optimizer = utils.opt_algorithm(self.finetune_params, self.learning_rate, self.alg_name)
        # self.optimizer = optim.SGD(params=self.finetune_params, lr=1e-3, momentum=0.9)

        self.data_root = opt.data_root
        self.train_list_file = opt.train_list_file
        self.val_list_file = opt.val_list_file
        self.label_list_file = opt.label_list_file

        self.save_model_step = opt.save_model_step
        self.save_model_path = opt.save_model_path
        self.save_model_name = opt.save_model_name


    def load_data(self):
        # 数据预处理
        img_transform = transforms.Compose([
            transforms.Resize(int(self.img_size * 1.1)),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        data_folder = {
            'train': myImageFloder(self.data_root, self.train_list_file, transform=img_transform),
            'val': myImageFloder(self.data_root, self.val_list_file, transform=img_transform)
        }

        self.dataloader = {
            'train':
                DataLoader(
                    data_folder['train'],
                    batch_size=self.train_batch_size,
                    shuffle=True,
                    num_workers=4),
            'val':
                DataLoader(
                    data_folder['val'],
                    batch_size=self.val_batch_size,
                    shuffle=False,
                    num_workers=4)
        }
        self.train_num = len(data_folder["train"])
        self.val_num = len(data_folder["val"])
        self.label_list = open(self.label_list_file, "r").readlines()
        self.class_num = len(self.label_list)



    def train_model(self):
        self.model.train()
        running_loss = 0.0
        running_acc = 0.0
        start_time = time.time()
        for i, data in enumerate(self.dataloader["train"], 1):
            img, label = data
            if self.label_th:
                label = utils.label_to_vector(label, self.class_num)
            if torch.cuda.is_available():
                img = Variable(img).cuda()
                label = Variable(label).cuda()

            else:
                img = Variable(img)
                label = Variable(label)

            # 向前传播
            out = self.model(img)
            loss = self.criterion(out, label)
            running_loss += loss.data[0] * label.size(0)
            _, pred = torch.max(out, 1)
            if self.label_th:
                __, re_label = torch.max(label, 1)
            else:
                re_label = label
            num_correct = (pred == re_label).int().sum()
            running_acc += num_correct.data[0]
            # 向后传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % 10 == 0:
                print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                    i,
                    len(self.dataloader["train"]), running_loss / (self.train_batch_size * i), running_acc
                                       / (self.train_batch_size * i)))
                print ('Average train speed: {:.6f} s/batch'.format((time.time()-start_time)/i))
        running_loss = running_loss / (self.train_num)
        running_acc = running_acc / (self.train_num)
        return running_loss, running_acc


    def test_model(self):
        self.model.eval()
        eval_loss = 0.0
        eval_acc = 0.0
        for data in self.dataloader["val"]:
            img, label = data
            if self.label_th:
                label = utils.label_to_vector(label, self.class_num)
            if torch.cuda.is_available():
                img = Variable(img, volatile=True).cuda()
                label = Variable(label, volatile=True).cuda()
            else:
                img = Variable(img, volatile=True)
                label = Variable(label, volatile=True)
            out = self.model(img)
            loss = self.criterion(out, label)
            eval_loss += loss.data[0] * label.size(0)
            _, pred = torch.max(out, 1)
            if self.label_th:
                __, re_label = torch.max(label, 1)
            else:
                re_label = label
            num_correct = (pred == re_label).int().sum()
            eval_acc += num_correct.data[0]
        eval_loss = eval_loss / (self.val_num)
        eval_acc = eval_acc / (self.val_num)
        return eval_loss, eval_acc


    def train_all(self):
        self.load_data()
        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []

        for epoch in range(self.num_epoches):
            # 加入衰减的learning_rate方式进行训练
            if (epoch + 1) % self.step_size == 0:
                self.learning_rate *= self.learning_rate_dec
                self.optimizer = utils.opt_algorithm(self.finetune_params, self.learning_rate, self.alg_name)
                # self.optimizer = optim.SGD(params=self.finetune_params, lr=self.learning_rate, momentum=0.9)

            print('epoch {}'.format(epoch + 1))
            print('*' * 10)

            running_loss, running_acc = self.train_model()
            train_loss.append(running_loss)
            train_acc.append(running_acc)
            print('Finish {} epoch\nTrain Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch + 1, running_loss, running_acc))

            eval_loss, eval_acc = self.test_model()
            test_loss.append(eval_loss)
            test_acc.append(eval_acc)
            print('Val Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss, eval_acc))
            print('-------------------------------------')
            if ((epoch + 1) % self.save_model_step == 0) or ((epoch + 1) == self.num_epoches):
                if not os.path.exists(self.save_model_path):
                    os.mkdir(self.save_model_path)
                save_model_res = os.path.join("{}".format(self.save_model_path),"{}_{}.pth".format(self.save_model_name, epoch + 1))
                torch.save(self.model.state_dict(), save_model_res)

        return train_loss, train_acc, test_loss, test_acc
