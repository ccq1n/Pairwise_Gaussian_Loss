import tensorflow as tf
import os, sys
import numpy as np

from loss_function import *
import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

'''
选择网络模型
选择损失函数，配置损失函数的参数
不同数据集（下载、类别数）

训练模型、测试模型主函数
'''

def train(epoch):


def test(epoch):



if __name__ == '__main__':
