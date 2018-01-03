# -*- coding: utf-8 -*-
import os
from sklearn.metrics import confusion_matrix, precision_score
import numpy as np

class result_summary():
    def __init__(self, pred, true, score = None, score_th = None, labels = None):
        self.pred = pred
        self.true = true
        self.labels = labels
        if labels is None:
            self.Get_labels()

        self.score = score
        self.score_th = score_th
        if not (self.score is None or self.score_th is None):
            self.Pred_score_filter()
        self.confusion_mat = self.Get_Confusion_matrix()

    def Get_labels(self):
        true_label = np.ndarray.tolist(self.true)
        self.labels = list(set(true_label))
        return self.labels


    def Pred_score_filter(self):
        filter_index = self.score < self.score_th
        self.pred[filter_index] = 0
        self.score[filter_index] = 0.0

    def Gwt_diag(self):
        return np.diag(self.Get_Confusion_matrix())


    # def Get_FP(self):
    def Get_Confusion_matrix(self):
        return confusion_matrix(self.true, self.pred, self.labels)

    def Get_TP(self):
        tp_list = np.diag(self.confusion_mat)
        tp_dict = {}
        for index_i, label_i in enumerate(self.labels):
            tp_dict[label_i] = tp_list[index_i]
        return tp_dict

    def Get_FP(self):
        fp_list = np.sum(self.confusion_mat, 0) - np.diag(self.confusion_mat)
        fp_dict = {}
        for index_i, label_i in enumerate(self.labels):
            fp_dict[label_i] = fp_list[index_i]
        return fp_dict

    def Get_FN(self):
        fn_list = np.sum(self.confusion_mat, 1) - np.diag(self.confusion_mat)
        fn_dict = {}
        for index_i, label_i in enumerate(self.labels):
            fn_dict[label_i] = fn_list[index_i]
        return fn_dict

    def Get_Accuracy(self):
        All_num = np.sum(self.confusion_mat)
        Ture_num = np.sum(np.diag(self.confusion_mat))
        return 1.0*Ture_num/All_num

    def Get_Recall(self):
        tp_list = np.diag(self.confusion_mat)
        recall_list = 1.0*tp_list/np.sum(self.confusion_mat, 1)
        recall_dict = {}
        for index_i, label_i in enumerate(self.labels):
            recall_dict[label_i] = recall_list[index_i]
        return recall_dict

    def Get_Precision(self):
        # precision_list = precision_score(self.true, self.pred, labels=self.labels, average=None)

        tp_list = np.diag(self.confusion_mat)
        precision_list = 1.0*tp_list/np.sum(self.confusion_mat, 0)
        precision_dict = {}
        for index_i, label_i in enumerate(self.labels):
            precision_dict[label_i] = precision_list[index_i]
        return precision_dict

    def Get_AP(self):
        if self.score is None:
            return None
        pred_dict = {}
        sorted_ind = np.argsort(-self.score)
        sort_pred = self.pred[sorted_ind]
        sort_true = self.true[sorted_ind]
        for index_i, pred_i in enumerate(sort_pred):
            if not pred_i in pred_dict:
                pred_dict[pred_i] = []
            if pred_i == sort_true[index_i]:
                pred_dict[pred_i].append(1.0)
            else:
                pred_dict[pred_i].append(0.0)

        pred_total_num = np.sum(self.confusion_mat, 0)
        true_total_num = np.sum(self.confusion_mat, 1)

        AP_dict = {}
        for i, label_i in enumerate(self.labels):
            pred_num_i = pred_total_num[i]
            true_num_i = true_total_num[i]

            tp_i = np.array(pred_dict[label_i])
            tp_i = np.cumsum(tp_i)

            recall_i = tp_i/np.maximum(true_num_i, np.finfo(np.float64).eps)
            precision_i = tp_i/np.maximum(pred_num_i, np.finfo(np.float64).eps)
            AP_dict[label_i] = self.voc_ap(recall_i, precision_i)
        return AP_dict

    def voc_ap(self, rec, prec):
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap