# -*- coding: utf-8 -*-
"""
仅做测试
"""

import matlab.engine
import util
import numpy as np


def test():
    eng = matlab.engine.start_matlab()
    list = eng.get_MRCG('\data\female_dev.wav')[0]
    print('list is',list)

if __name__ == "__main__":
    test()
