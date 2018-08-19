import os
import keras.backend as K

__author__ = 'roeiherz'

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    tt = K.get_session()
    while True:
        continue
