#! /usr/bin/env python3
import os
rerun_num = 105

for x in range(rerun_num):
    os.system("python train.py")
    print("****************************")
    print("****************************")
    print("****************************")
    print("程序将重启第: % d 次..." % (x+1))
