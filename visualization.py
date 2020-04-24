#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:xuliheng time:2019/11/21
import csv
import matplotlib.pyplot as plt

with open('train_history.csv') as f:
    # 创建一个阅读器：将f传给csv.reader
    reader = csv.reader(f)
    # 使用csv的next函数，将reader传给next，将返回文件的下一行
    header_row = next(reader)
    # 读取res_map
    # 创建res_map的列表
    res_mAPs = []

    for row in reader:
        if len(row) > 0:
            res_mAP = float(row[1])
            res_mAPs.append(res_mAP)

print(res_mAPs)
print(type(res_mAPs[0]))
# 设置图片大小
fig = plt.figure(dpi=128, figsize=(14, 6))
plt.plot(res_mAPs, c='red', linewidth=1)  # 设置颜色、线条粗细

# 设置图片格式
# plt.title('Res_mAP', fontsize=24)  # 标题
# plt.xlabel('Number of sampled scene', fontsize=14)
# plt.ylabel('', fontsize=14)
plt.title('Controller reward', fontsize=24)  # 标题
plt.xlabel('Number of sampled scene', fontsize=14)
plt.ylabel('', fontsize=14)
plt.show()  # 输出图像

