#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:xuliheng time:2019/12/9
from manager import RewardManager
import csv


manager = RewardManager(clip_rewards=0.0,
                        acc_beta=0.8)
for trial in range(1401):
    buffer = []
    buffer.append(trial)
    reward, previous_res_mAP = manager.get_reward(buffer)
    print("Rewards : ", reward, "Res_mAP : ", previous_res_mAP)

    with open('pictur_performance.csv', 'a', newline='') as f:
        data = [trial, previous_res_mAP]
        writer = csv.writer(f)
        writer.writerow(data)
