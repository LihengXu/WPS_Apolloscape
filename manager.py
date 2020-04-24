#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:xuliheng time:2019/11/20
import pickle
import numpy as np
from utils.eval_utils import voc_eval, f1_eval
from utils.misc_utils import read_class_names, AverageMeter


class RewardManager:
    def __init__(self,
                 class_name_path="./data/class.names",
                 acc_beta=0.8, clip_rewards=0.0
                 ):
        # some path
        self.class_name_path = class_name_path
        # some numbers
        self.use_voc_07_metric = False
        # some params
        self.classes = read_class_names(self.class_name_path)
        self.class_num = len(self.classes)
        # reward params
        self.clip_rewards = clip_rewards
        self.beta = acc_beta
        self.beta_bias = acc_beta
        self.moving_res_mAP = 0.0

    def get_reward(self, action):
        index = action
        dict = {}
        preds = []

        with open('./data/Apolloscape_eval.pkl', 'rb') as f:
            gt_dict = pickle.load(f)
            val_preds = pickle.load(f)
        # with open('./data/coco_eval.pkl', 'rb') as f:
        #     gt_dict = pickle.load(f)
        #     val_preds = pickle.load(f)

        # The predictions of model from detectron2
        # with open('./data/detectron2_results/detectron2_gt_dict.pkl', 'rb') as f:
        #     gt_dict = pickle.load(f)
        # with open('./data/detectron2_results/retinanet_R_50_FPN_1x.pkl', 'rb') as f:
        #     val_preds = pickle.load(f)

        for m in index:
            dict[m] = gt_dict[m]

        for n in val_preds:
            if (n[0] in index):
                preds.append(n)

# ---------------------------------------------------------------------------------------------------------------------
        # # compute the mAP
        # rec_total, prec_total, ap_total = AverageMeter(), AverageMeter(), AverageMeter()
        # for ii in range(self.class_num):
        #     npos, nd, rec, prec, ap = voc_eval(dict, preds, ii, iou_thres=0.5, use_07_metric=self.use_voc_07_metric)
        #     rec_total.update(rec, npos)
        #     prec_total.update(prec, nd)
        #     ap_total.update(ap, npos)
        #     print('Class {}: Recall: {:.4f}, Precision: {:.4f}, AP: {:.4f}'.format(ii, rec, prec, ap))
        # mAP = ap_total.average
        # res_mAP = 1 - mAP
        # print('final mAP: {:.4f}'.format(mAP))
        # print("recall: {:.3f}, precision: {:.3f}".format(rec_total.average, prec_total.average))

        # # compute the F1-score
        npos, nd, rec, prec, f1 = f1_eval(dict, preds, iou_thres=0.5)
        print('Recall: {:.4f}, Precision: {:.4f}, F1-score: {:.4f}'.format(rec, prec, f1))
        mAP = f1
        res_mAP = 1 - mAP
# ----------------------------------------------------------------------------------------------------------------------


        # compute the reward
        reward = (res_mAP - self.moving_res_mAP)

        # if rewards are clipped, clip them in the range -0.05 to 0.05
        if self.clip_rewards:
            reward = np.clip(reward, -0.05, 0.05)

        # update moving accuracy with bias correction for 1st update
        if self.beta > 0.0 and self.beta < 1.0:
            self.moving_res_mAP = self.beta * self.moving_res_mAP + (1 - self.beta) * res_mAP
            self.moving_res_mAP = self.moving_res_mAP / (1 - self.beta_bias)
            self.beta_bias = 0

            # reward *= 0.01

            reward = np.clip(reward, -0.1, 0.1)

        print()
        print("Manager: EWA res_mAP = ", self.moving_res_mAP)

        return reward, res_mAP

