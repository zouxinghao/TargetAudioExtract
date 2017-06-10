# -*- coding: utf-8 -*-

"""
各种工具方法，服务于各处理模块
"""
import matlab.engine
import logging
import csv
import os
import config
import soundfile
import random
import numpy as np
from resampy import resample
import numbers
import os


def read_csv(filename):
    """
    读取简单的csv文件，不带列标题，每行一个列表，元素均为str
    :param filename: csv文件路径
    :return: [['row1 col1', 'row1 col2',...],...]
    """
    with open(filename) as df:
        datareader = csv.reader(df)
        return list(datareader)


def load_data(data_list_file, dataset_dir):
    """
    读取csv文件，并加载指定的m个wav文件。csv文件第一列为wav文件名，第二列0/1标记是否故障。
    注意：音频采样率如果与配置不同，会使用重采样方法保证一致。
    :param data_list_file: csv文件路径
    :param dataset_dir: wav文件目录
    :return: audios: m个音频list
            labels: m个 0/1 故障标记
    """
    data = read_csv(data_list_file)
    audios = []
    for d in data:
        audios.append(read_wav(dataset_dir + "/" + d[0]))
    return audios


def load_data_from_matlab(data_list_file, dataset_dir):
    data = read_csv(data_list_file)
    audios = []
    eng = matlab.engine.start_matlab()
    for d in data:
        print('enter audio append')
        audios.append(np.array(eng.get_MRCG(dataset_dir + "/" + d[0])[0]))
        print('audios is', audios)
    return audios


def create_more_signals(signal_path, data_root, data_list):
    """
     这个函数主要用来制造更多的原信号
     filepath是文件的路径名
    """
    data = read_wav(signal_path)
    # 获得文件名，如male_01
    filename = os.path.basename(signal_path)
    filename = filename.split('.')[0]
    with open(data_list, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in np.arange(1, 10, 0.2):
            # 将数据的前后按比例的调换，形成新的数据
            data_1 = data[0:int(i / 10 * len(data))]
            data_2 = data[int(i / 10 * len(data)):len(data)]
            data = np.concatenate((data_2, data_1))

            fname = '%s%g.wav' % (filename, i)
            soundfile.write(data_root+'/'+fname, data, config.SAMPLE_RATE)
            writer.writerow([fname])


def read_wav(filename):
    """
    读取wav文件。
    注意：音频采样率如果与配置不同，会使用重采样方法保证一致。
    :param filename: wav文件路径
    :return: audio: 音频信号list
    """
    audio, samplerate = soundfile.read(filename)
    if samplerate != config.SAMPLE_RATE:
        audio = resample(audio, samplerate, config.SAMPLE_RATE)
        logging.warning("%s samplerate convert from %gHZ to %gHZ" %
                        (filename, samplerate, config.SAMPLE_RATE))
    return audio


def read_wav_from_matlab(filename):
    eng = matlab.engine.start_matlab()
    audio = eng.get_MRCG(filename)[0]
    return np.array(audio)


def r_abs(X):
    """
    将所有数值取绝对值
    """
    if hasattr(X, "__len__"):
        x_list = []
        for e in X:
            x_list.append(r_abs(e))
        return x_list
    else:
        return abs(X)


def mix(sig_one, sig_two, output=None):
    """
    将音频信号按指定信噪比混合，如需要同时保存到文件。
    可以传入一段音频信号，或是几段音频信号
    :param sig_one: [...] 或 [[...]...]
    :param sig_two: 与sig_one对应
    :param output: 输出文件名
    :return: 源信号，噪音，混合信号
    """
    '''
    如果输入list元素为数字，则判断输入一段信号，否则，判断输入多段信号。
    对一段信号，先将源信号与噪音对齐（便之等长），再
    对多段信号，递归调用进行处理。
    '''
    if isinstance(sig_one[0], numbers.Number):
        # 输入一段信号
        # signal one 与 signal two 长度可能不等，首先使之等长
        if len(sig_one) > len(sig_two):
            start = random.randint(0, len(sig_one) - len(sig_two))
            sig_one = sig_one[start:(start + len(sig_two))]
        elif len(sig_one) < len(sig_two):
            start = random.randint(0, len(sig_two) - len(sig_one))
            sig_two = sig_two[start:(start + len(sig_one))]
        sig_one = np.array(sig_one)
        sig_two = np.array(sig_two)

        # TODO deal with number overflow

        # 混合
        mixture = sig_one + sig_two

        # 保存到文件
        if output is not None:
            soundfile.write(output + '.mix.wav', mixture, config.SAMPLE_RATE)
            soundfile.write(output + '.sig1.wav', sig_one, config.SAMPLE_RATE)
            soundfile.write(output + '.sig2.wav', sig_two, config.SAMPLE_RATE)

        return sig_one, sig_two, mixture

    # 输入多段信号的情况
    sig_one_list, sig_two_list, mix_list = [], [], []
    for i in range(len(sig_one)):
        if output is None:
            s, n, m = mix(sig_one[i], sig_two[i], None)
        else:
            s, n, m = mix(sig_one[i], sig_two[i], output[i])
        sig_one_list.append(s)
        sig_two_list.append(n)
        mix_list.append(m)
    return sig_one_list, sig_two_list, mix_list


def mkdir_p(dir_path):
    '''
    mkdri -p，创建目录，不存在则创建，如需要也创建上级目录
    :param dir_path: 目录路径
    '''
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
