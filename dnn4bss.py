# -*- coding: utf-8 -*-
"""
基于神经网络的盲源分离模型

参考：
P.-S. Huang, M. Kim, M. Hasegawa-Johnson, P. Smaragdis, "Deep Learning for
Monaural Speech Separation,"
in IEEE International Conference on Acoustic,
Speech and Signal Processing 2014.
"""

import matlab.engine
import config
import logging
import util
import os
import numpy as np
import nn
import soundfile as sf
import stft


def train():
    logging.info('train bss model')
    sig1_list = util.load_data(config.DATA_TRAIN_LIST, config.DATA_TRAIN_ROOT)
    # sig1_list = util.load_data_from_matlab(config.DATA_TRAIN_LIST, config.DATA_TRAIN_ROOT)
    sig2 = util.read_wav(config.DATA_TRAIN_STATIC_SIGNAL)
    # sig2 = util.read_wav_from_matlab(config.DATA_TRAIN_STATIC_SIGNAL)
    # print('sig1_list is', sig1_list)
    # print('sig2 is', sig2)
    if config.NORMALIZATION:
        sig1_list = [_normalize(sig) for sig in sig1_list]
        sig2 = _normalize(sig2)

    logging.info('mix signals')
    util.mkdir_p(config.DATA_TRAIN_MIX_ROOT)
    mix_filenames = _gen_mix_filename(config.DATA_TRAIN_LIST,
                                      config.DATA_TRAIN_STATIC_SIGNAL,
                                      config.DATA_TRAIN_MIX_ROOT)
    sig2_list = [sig2 for _ in range(len(sig1_list))]
    sig1_list, sig2_list, mix_list = util.mix(sig1_list, sig2_list,
                                              output=mix_filenames)

    # print('sig1_list is', sig1_list)
    # print('sig2_list is', sig2_list)
    # print('mix_list is', mix_list)
    # logging.info('extract stft features')
    # sig1_stft = _extract_stft(sig1_list)
    # print('sig1_stft', sig1_stft)
    # sig2_mrcg = _extract_stft(sig2_list)
    # mix_mrcg = _extract_stft(mix_list)

    logging.info('extract mrcg features')
    sig1_mrcg = _extract_mrcg(sig1_list)
    print('sig1_mrcg is', sig1_mrcg)
    sig2_mrcg = _extract_mrcg(sig2_list)
    print('sig2_mrcg is', sig2_mrcg)
    mix_mrcg = _extract_mrcg(mix_list)
    print('mix_mrcg is', mix_mrcg)

    logging.info('train neural network')
    sig1_mrcg = util.r_abs(sig1_mrcg)
    sig2_mrcg = util.r_abs(sig2_mrcg)
    mix_mrcg = util.r_abs(mix_mrcg)
    train_x, train_y = mix_mrcg, []
    for s, n in zip(sig1_mrcg, sig2_mrcg):  # 将signal和noise拼接作为nn输出
        train_y.append(np.concatenate((s, n), axis=1))
    train_x = [_extend(sig, config.EXTEND_NUM) for sig in train_x]

    # 各层神经元数量
    layer_size = [len(train_x[0][0])]  # 输入层
    layer_size.extend(config.LAYER_SIZE)  # 隐藏层
    layer_size.append(len(train_y[0][0]))  # 输出层

    dnn = nn.NNet(layer_size)
    util.mkdir_p(config.MODEL_ROOT)
    dnn.train(train_x, train_y, model_path=config.MODEL_NN_PATH,
              training_epochs=config.EPOCH, learning_rate=config.LEARNING_RATE)
    # dnn.test(train_x, train_y, model_path=config.MODEL_DN_NN_PATH)


def _normalize(sig):
    sig = np.array(sig)
    return sig / np.sqrt(np.sum(sig ** 2))


def _extend(sig, increment):
    """
    将每帧与前后帧连结
    :param sig: 信号
    :param increment: 前/后连结帧数量
    :return:
    """
    recurrent_train_x = []
    for i in range(len(sig)):  # 帧数
        temp = []
        for k in range(i - increment, i + increment + 1):
            if k < 0 or k >= len(sig):
                temp.extend(np.zeros(len(sig[i])))
            else:
                temp.extend(sig[k])
        temp_2 = np.array(temp)
        recurrent_train_x.append(temp_2)
    return recurrent_train_x


def _extract_stft(audios):
    stfts = []
    for ad in audios:
        st = stft.spectrogram(ad, framelength=config.STFT_POINT,
                              overlap=config.STFT_OVERLAP)
        st = st.transpose()
        stfts.append(st)
    return stfts


def _extract_mrcg(audios):
    """the function extract MRCG features"""
    mrcgs = []
    eng = matlab.engine.start_matlab()
    print('start extract')
    for ad in audios:
        ad = matlab.double(ad.tolist())
        mrcg = eng.MRCG_features(ad, config.SAMPLE_RATE)
        mrcgs.append(np.array(mrcg))
        # print('mrcgs is', mrcgs)
    print('extract finished')
    print('mrcgs_2 is', mrcgs)
    return mrcgs


def test():
    logging.info('test denoising model')
    logging.info("read test signal according to %s, read noise from %s" %
                 (config.DATA_TEST_LIST, config.DATA_TEST_STATIC_SIGNAL))
    sig1_list = util.load_data(config.DATA_TEST_LIST, config.DATA_TEST_ROOT)
    sig2 = util.read_wav(config.DATA_TEST_STATIC_SIGNAL)

    if config.NORMALIZATION:
        sig1_list = [_normalize(sig) for sig in sig1_list]
        sig2 = _normalize(sig2)

    logging.info('mix signals')
    util.mkdir_p(config.DATA_TEST_MIX_ROOT)
    mix_filenames = _gen_mix_filename(config.DATA_TEST_LIST,
                                      config.DATA_TEST_STATIC_SIGNAL,
                                      config.DATA_TEST_MIX_ROOT)
    sig2_list = [sig2 for _ in range(len(sig1_list))]
    sig1_list, sig2_list, mix_list = util.mix(sig1_list,
                                              sig2_list, output=mix_filenames)

    logging.info('extract stft features')
    mix_mrcg = _extract_mrcg(mix_list)

    mix_data = util.r_abs(mix_mrcg)
    mix_data = [_extend(sig, config.EXTEND_NUM) for sig in mix_data]

    logging.info('run neural network')
    # 各层神经元数量
    layer_size = [len(mix_data[0][0])]  # 输入层
    layer_size.extend(config.LAYER_SIZE)  # 隐藏层
    layer_size.append(len(mix_mrcg[0][0]) * 2)  # 输出层
    dnn = nn.NNet(layer_size)
    sig1_sig2 = dnn.run(mix_data, model_path=config.MODEL_NN_PATH)
    # 神经网络输出为预测signal1与signal2的拼接，此处分离
    sig1_mrcg, sig2_mrcg = _separate(sig1_sig2)

    logging.info('Time-Frequency Masking')
    mask = []
    for s, n in zip(sig1_mrcg, sig2_mrcg):
        mask.append(_build_ibm(s, n))
    sig1_mrcg = [np.multiply(mix, m) for mix, m in zip(mix_mrcg, mask)]
    sig2_mrcg = [np.subtract(mix, sig) for mix, sig in zip(mix_mrcg, sig1_mrcg)
                 ]
    logging.info('Inverse Short-Time Fourier Transformation')
    # 从频域转换到时域
    sep_sig1 = _istft(sig1_mrcg)
    sep_sig2 = _istft(sig2_mrcg)

    logging.info("write test audio into dir %s" % config.DATA_TEST_MIX_ROOT)
    util.mkdir_p(config.DATA_TEST_MIX_ROOT)
    file_list = _gen_mix_filename(config.DATA_TEST_LIST,
                                  config.DATA_TEST_STATIC_SIGNAL,
                                  config.DATA_TEST_MIX_ROOT)
    for ss, sn, f in zip(sep_sig1, sep_sig2, file_list):
        sf.write(f + '.sep.sig1.wav', ss, config.SAMPLE_RATE)
        sf.write(f + '.sep.sig2.wav', sn, config.SAMPLE_RATE)

    logging.info('run bss evaluation')
    sdr, sir, sar = _bss_eval()
    logging.info('SDR: %g, SIR: %g, SAR: %g' % (sdr, sir, sar))


def _build_ibm(sig1, sig2):
    # 创建 ideal binary mask
    # TODO deal with snr != 0
    ibm = []
    for s1, s2 in zip(sig1, sig2):
        s1, s2 = np.abs(s1), np.abs(s2)
        # bool object has no astype attribute
        i = (s1 > s2).astype(np.float64)
        ibm.append(i)
    return ibm


def _istft(stft_matrix_list):
    '''
    Inverse Short-Time Fourier Transformation
    '''
    audios = []
    for sm in stft_matrix_list:
        sm = np.transpose(sm)
        ad = stft.ispectrogram(sm, framelength=config.STFT_POINT,
                               overlap=config.STFT_OVERLAP)
        audios.append(ad)
    return audios


def _bss_eval():
    '''
    通过调用Matlab BSS_EVAL工具进行表现评估
    :return: sdr、sir、sar三个指标均值
    '''
    file_list = _gen_mix_filename(config.DATA_TEST_LIST,
                                  config.DATA_TEST_STATIC_SIGNAL,
                                  config.DATA_TEST_MIX_ROOT)
    logging.info('connect to matlab')

    avg_sdr, avg_sir, avg_sar = 0., 0., 0.
    eng = matlab.engine.start_matlab()

    for filepath in file_list:
        sig1_wav = filepath + '.sig1.wav'
        sig2_wav = filepath + '.sig2.wav'
        sep_sig1_wav = filepath + '.sep.sig1.wav'
        sep_sig2_wav = filepath + '.sep.sig2.wav'
        mix_wav = filepath + '.mix.wav'
        # matlab执行bss_eval.m
        sdr, sir, sar = eng.bss_eval(sig1_wav, sig2_wav, sep_sig1_wav,
                                     sep_sig2_wav, mix_wav, nargout=3)
        avg_sdr += sdr / len(file_list)
        avg_sir += sir / len(file_list)
        avg_sar += sar / len(file_list)

    return avg_sdr, avg_sir, avg_sar


def _separate(X_list):
    # 把 X_list 的每行均分成两半
    a, b = [], []
    for X in X_list:
        a.append([row[:int(len(row) / 2)] for row in X])
        b.append([row[int(len(row) / 2):] for row in X])

    return a, b


def _gen_mix_filename(sig1_list_csv, sig2_path, mix_dir):
    '''
    生成混合音频文件的路径str
    :param sig1_list_csv: csv文件，第一列记录音频文件名
    :param sig2_path: 另一个音源文件目录
    :param mix_dir: 混合音频保存目录
    :return:
    '''
    mix_file_list = []
    signals_file_list = util.read_csv(sig1_list_csv)
    sig2_fillename = os.path.basename(sig2_path).split('.')[0]
    for sig1_file in signals_file_list:
        sig1_filename = os.path.splitext(sig1_file[0])[0]
        mix_file_list.append('%s/%s+%s' %
                             (mix_dir, sig1_filename, sig2_fillename))
    return mix_file_list


def log_config():
    with open(config.LOG_FILENAME, 'a', encoding='utf-8') as log:
        log.write('config.py-----------------------------------------------\n')
        with open('config.py', 'r', encoding='utf-8') as cf:
            for line in cf:
                log.write(line)
        log.write('\n\n')
        log.write('%s---------------------------------------------------\n'
                  % config.RUN_SCRIPT)
        with open(config.RUN_SCRIPT, 'r', encoding='utf-8') as scp:
            for line in scp:
                log.write(line)
        log.write('\n\n')


if __name__ == "__main__":
    # log_config()

    logging.info('create train loop signals')
    util.mkdir_p(config.DATA_TRAIN_ROOT)
    util.create_more_signals(config.DATA_TRAIN_LOOP_SIGNAL,
                             config.DATA_TRAIN_ROOT, config.DATA_TRAIN_LIST)
    train()

    # logging.info('create test loop signals')
    # util.mkdir_p(config.DATA_TEST_ROOT)
    # util.create_more_signals(config.DATA_TEST_LOOP_SIGNAL,
    #                          config.DATA_TEST_ROOT,
    #                          config.DATA_TEST_LIST)
    # test()
