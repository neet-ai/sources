#! -*- coding: utf-8 -*-
import glob
import os.path as op
import numpy as np

from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.preprocessing.image import random_rotation, random_shift, random_zoom
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.regularizers import l2, activity_l2
from keras.utils import np_utils


################################
###### 画像データの前処理 ######
################################
def preprocess(dirname, filename, height=64, width=64, var_kind=7, var_amount=8):
    flag1 = False; flag2 = False; flag3 = False;
    if var_kind % 2 == 1:
        flag1 = True
    if (var_kind // 2) % 2 == 1:
        flag2 = True
    if var_kind // 4 == 1:
        flag3 = True
    
    num = 0
    arrlist = []
    myadd = arrlist.append
    files = glob.glob(dirname + "/*.jpg")
    for imgfile in files:
        img = load_img(imgfile, target_size=(height, width))
        array = img_to_array(img) / 255
        myadd(array)
        if var_kind != 0:
            for i in range(var_amount-1):
                arr2 = array
                if flag1:
                    arr2 = random_rotation(arr2, rg=360)
                if flag2:
                    arr2 = random_shift(arr2, wrg=0.1, hrg=0.1)
                if flag3:
                    arr2 = random_zoom(arr2, zoom_range=(0.5, 1.0))
                myadd(arr2)
        num += 1
        
    nplist = np.array(arrlist)
    np.save(filename, nplist)
    print(str(nplist))
    print(">> " + dirname + "から" + str(num) + "個のファイル読み込み成功")
    return nplist


################################
######### モデルの構築 #########
################################
def build_deep_cnn(ipshape=(32, 32, 3), num_classes=3):
    model = Sequential()

    model.add(Convolution2D(96, 3, 3, border_mode='same', input_shape=ipshape))
    model.add(Activation('relu'))

    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(256, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(256, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))    
    
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    return model


################################
############# 学習 #############
################################
def learning(tsnum=34, nb_epoch=50, batch_size=8):
    X_TRAIN_list = []; Y_TRAIN_list = []; X_TEST_list = []; Y_TEST_list = [];
    target = 0
    while True:
        while True:
            filename = input("読み込むnumpyファイル(「END」で次の処理へ)：")
            if op.isfile(filename) or filename == "END":
                break
            print("そのファイルは存在しません！")
        if filename == "END":
            break
        data = np.load(filename)
        trnum = data.shape[0] - tsnum
        X_TRAIN_list += [data[i] for i in range(trnum)]
        Y_TRAIN_list += [target] * trnum
        X_TEST_list  += [data[i] for i in range(trnum, trnum+tsnum)]
        Y_TEST_list  += [target] * tsnum;
        target += 1

    X_TRAIN = np.array(X_TRAIN_list + X_TEST_list)
    Y_TRAIN = np.array(Y_TRAIN_list + Y_TEST_list)
    print(">> 学習サンプル数 : ", X_TRAIN.shape)
    y_train = np_utils.to_categorical(Y_TRAIN, target+1)
    
    # 学習率の変更
    class Schedule(object):
        def __init__(self, init=0.001):
            self.init = init
        def __call__(self, epoch):
            lr = self.init
            for i in range(1, epoch+1):
                if i%5==0:
                    lr *= 0.5
            return lr
    
    def get_schedule_func(init):
        return Schedule(init)
    
    lrs = LearningRateScheduler(get_schedule_func(0.001))
    model = build_deep_cnn(ipshape=(X_TRAIN.shape[1], X_TRAIN.shape[2], X_TRAIN.shape[3]), num_classes=target+1)
    
    print(">> 学習開始")
    hist = model.fit(X_TRAIN, y_train,
                     batch_size=batch_size,
                     verbose=1,
                     nb_epoch=nb_epoch,
                     validation_split=0.1,
                     callbacks=[lrs])


################################
############ メイン ############
################################
yesno = input(">> 画像データをnumpyデータに変換しますか？(y/n) : ")
if yesno.lower() == "y" or yesno.lower() == "yes":
    print(">> 拡張子が.jpgのファイルのみ読み込みます。")
    while True:
    
        # ディレクトリ名入力
        while True:
            dirname = input(">> 画像のあるディレクトリ(「END」で終了)：")
            if op.isdir(dirname) or dirname == "END":
                break
            print(">> そのディレクトリは存在しません！")
        if dirname == "END":
            break
            
        filename = input(">> 保存ファイル名：")
            
        # 関数実行
        preprocess(dirname, filename, height=32, width=32, var_kind=1, var_amount=3)
        
yesno = input(">> 学習を実行しますか？(y/n) : ")
if yesno.lower() == "y" or yesno.lower() == "yes":
    learning(tsnum=34, nb_epoch=50, batch_size=8)
