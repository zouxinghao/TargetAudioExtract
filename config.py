# -*- coding: utf-8 -*-
"""
运行参数配置，包括日志，数据文件，模型文件，模型参数等
"""

import logging
import inspect
import os

################################
# 日志配置
_run_script = inspect.stack()[-1][1]
# 运行脚本名
RUN_SCRIPT = os.path.basename(_run_script)
_run_script_no_ext = os.path.splitext(RUN_SCRIPT)[0]
# 日志文件名
LOG_FILENAME = '%s.log' % _run_script_no_ext

# 配置日志输出文件及级别
FORMAT = '%(asctime)s::%(levelname)s::%(message)s'
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO, format=FORMAT,
                    filemode='a')

# 同时输出到控制台
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter(FORMAT))
logging.getLogger('').addHandler(console)

################################
# 数据目录与文件路径
# 数据根目录
DATA_ROOT = './data'

# 训练数据
DATA_TRAIN_ROOT = DATA_ROOT + '/male_train'
# csv文件，第一列为训练wav文件名
DATA_TRAIN_LIST = DATA_TRAIN_ROOT + '/data.csv'
# 混合后数据目录
DATA_TRAIN_MIX_ROOT = DATA_ROOT + '/train.mix'

# 测试数据，参考训练数据
DATA_TEST_ROOT = DATA_ROOT + '/male_test'
DATA_TEST_LIST = DATA_TEST_ROOT + '/data.csv'
DATA_TEST_MIX_ROOT = DATA_ROOT + '/test.mix'

# 用于循环得到更多的信号
DATA_TRAIN_LOOP_SIGNAL = DATA_ROOT + '/male_train.wav'
DATA_TEST_LOOP_SIGNAL = DATA_ROOT + '/male_test.wav'

# 训练与测试信号
DATA_TRAIN_STATIC_SIGNAL = DATA_ROOT + '/female_train.wav'
DATA_TEST_STATIC_SIGNAL = DATA_ROOT + '/female_test.wav'

################################
# 模型存放目录与文件路径
MODEL_ROOT = './model'
# 神经网络模型
MODEL_NN_PATH = MODEL_ROOT + '/neural_net.model'

################################
# 训练参数配置
# wav采样率
SAMPLE_RATE = 16000
# STFT每帧点数
STFT_POINT = 1024
# STFT每帧重叠 1/DN_STFT_OVERLAP
STFT_OVERLAP = 2
# 神经网络隐藏层每层激活元数量
LAYER_SIZE = [16]
# 模型训练次数
EPOCH = 1000
# 神经网络输入，每帧前/后扩展数
EXTEND_NUM = 0
# 神经网络学习速率
LEARNING_RATE = 0.001
# 是否正则化
NORMALIZATION = True
