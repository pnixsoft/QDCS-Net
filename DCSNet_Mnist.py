#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/2 21:16
# @Time    : 2023/10/03 15:26 整理
# @Author  : JamesQ
# @Email   : pnixsoft@163.com
# @File    : DCSNet_Mnist.py
# @Software: PyCharm Community Edition
# Redistribution of this code is permitted.


import numpy as np
import argparse
# import matplotlib.pyplot as plt
import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Reshape, Flatten, Add,BatchNormalization, MaxPooling2D, UpSampling2D
from keras import backend as K
from keras import callbacks
from keras.callbacks import LearningRateScheduler
import datetime
import os
import scipy.io as scio
import tensorflow as tf
from keras.datasets import mnist

#============================================================================================
#  1  数据集准备， 载入数据。
#============================================================================================
def SAEModel(CR):
#============================================================================================
#  1  Encoder 网络模型
#============================================================================================
    input_img = Input(shape=(28, 28 ,1))  # adapt this if using `channels_first` image data format
    # "encoded" is the encoded representation of the input
    encoded = Conv2D(2,(3,3), activation='relu', padding = 'same')(input_img)
    encoded = Flatten()(encoded)
    encoded = Dense(CR, activation='sigmoid')(encoded)

#============================================================================================
#  2  Decoder 网络模型
#============================================================================================
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(784, activation='sigmoid')(encoded)
    # print('decoded shape:')
    # print(decoded.shape)
    decoded = Reshape((28,28,1))(decoded)
    # print(decoded.shape)

    decoded1 = Conv2D(2, (3, 3), activation='relu', padding='same')(decoded)
    decoded = BatchNormalization()(decoded1)
    decoded = Conv2D(8, (3, 3), activation='relu', padding='same')(decoded1)
    decoded = BatchNormalization()(decoded)
    decoded = Conv2D(16, (3, 3), activation='relu', padding='same')(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Conv2D(2, (3, 3), activation='relu', padding='same')(decoded)
    decoded = BatchNormalization()(decoded)
    # decoded = decoded+decoded1
    decoded = Add()([decoded1, decoded])

    decoded1 = Conv2D(2, (3, 3), activation='relu', padding='same')(decoded)
    decoded = BatchNormalization()(decoded1)
    decoded = Conv2D(8, (3, 3), activation='relu', padding='same')(decoded1)
    decoded = BatchNormalization()(decoded)
    decoded = Conv2D(16, (3, 3), activation='relu', padding='same')(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Conv2D(2, (3, 3), activation='relu', padding='same')(decoded)
    decoded = BatchNormalization()(decoded)
    # decoded = decoded+decoded1
    decoded = Add()([decoded1, decoded])


    decoded1 = Conv2D(2, (3, 3), activation='relu', padding='same')(decoded)
    decoded = BatchNormalization()(decoded1)
    decoded = Conv2D(8, (3, 3), activation='relu', padding='same')(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Conv2D(16, (3, 3), activation='relu', padding='same')(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Conv2D(1, (3, 3), activation='relu', padding='same')(decoded)
    decoded = BatchNormalization()(decoded)

    autoencoder = Model(input_img, decoded)

    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)
    return autoencoder,encoder


def LoadData():

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    X_train = X_train[:, :, :, None]  # 扩维： (60000,28,28)--> (60000,28,28,1)
    X_test = X_test[:, :, :, None]
    train_x = X_train.reshape(len(X_train), 28, 28, 1)
    test_x = X_test.reshape(len(X_test), 28, 28, 1)

    return train_x,test_x

#============================================================================================
#  4  网络模型用于预测，并保存测试结果到文件
#============================================================================================
def Predict(autoencoder, encoder, test_x):
    encoded_imgs = encoder.predict(test_x)
    decoded_imgs = autoencoder.predict(test_x)

    test_x_filename ='CSImg_x_test.mat'
    Encoded_filename ='CSImg_ecoded.mat'
    Decoded_filename ='CSImg_decoded.mat'

    scio.savemat(test_x_filename,{'x':test_x})
    scio.savemat(Encoded_filename,{'e':encoded_imgs})
    scio.savemat(Decoded_filename,{'d':decoded_imgs})
    print('Save prediction OK!')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train parameters setting")
    parser.add_argument('-e','--epochs', dest='epochs', help="Training number of epochs", type=int, default=30)
    parser.add_argument('-b','--batchsize', dest='batchsize', help="batchsize", type=int, default=256)
    parser.add_argument('-cr','--compressratio', dest='compressratio', help="compressratio", type=int, default=256)
    parser.add_argument('-lr','--learningrate', dest='learningrate', help="learningrate", type=int, default=0)
    args = parser.parse_args()


    train_x,test_x = LoadData()

    SAE,encoder = SAEModel(args.compressratio)
    print(SAE.summary())
    SAE.compile(optimizer='adadelta', loss='mse')


    SAE.fit(train_x, train_x,
            epochs=args.epochs,
            batch_size=args.batchsize,
            shuffle=True,validation_data=(test_x,test_x))
   
    Predict(SAE, encoder, test_x)





















