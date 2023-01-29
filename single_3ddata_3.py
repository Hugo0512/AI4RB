# USAGE
# python mixed_training.py --dataset Houses-dataset/Houses\ Dataset/
import io
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# import the necessary packages
import sys

import numpy as np
# import tensorflow as tf
from keras.utils import to_categorical
from keras.utils import Sequence
from keras import optimizers
from keras_applications.resnet import ResNet101,ResNet50,ResNet152
from keras.layers import Dense,Flatten,GlobalAveragePooling2D,BatchNormalization,GlobalMaxPooling2D,Input,Conv2D, SeparableConv2D,Activation,AveragePooling2D,Dropout
from keras.utils.training_utils import multi_gpu_model
from keras import backend as K
import os
from keras.models import Model
import keras
from keras.callbacks import ModelCheckpoint,EarlyStopping
# from keras.layers import Dense,Flatten,GlobalAveragePooling2D,BatchNormalization,GlobalMaxPooling2D,Input,Conv2D, SeparableConv2D,Activation,AveragePooling2D,Dropout
from keras_applications.resnet import ResNet101,ResNet50,ResNet152
from keras.applications import DenseNet121,InceptionResNetV2,InceptionV3
from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
# import models
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.layers import concatenate
import numpy as np
import csv
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import cv2
class DataGenerator(Sequence):
    """
    基于Sequence的自定义Keras数据生成器
    """



    def __init__(self,data_dir,patient_list_txtfile,batch_size,shuffle=True):
    # def __init__(self):

        """ 初始化方法
        :param splitflag；区分train还是validation
        :param patient_list_txtfile,save the txt file of sample order,such as train1.txt test1.txt
        :param shuffle: 每一个epoch后是否打乱数据
        :param batch_size: 每一个epoch中clips的个数
        :param label_file_csv: the label of each sample
        :param secondary_label_filename:去掉了不能用的样本的标签文件名
        """
        #read txt file to obtain all sample

        # labels = pandas.read_csv(label_file_csv)
        # patient_list_txtfile = patient_list_txtfile

        temp = [];
        fid = open(patient_list_txtfile, 'r', encoding='utf-8')
        templabels = []
        deleteindex3 = []
        lines = fid.readlines()
        for index in range(len(lines)):
            line = lines[index]

            position = line.find(" ")
            temp.append(line[0:position])
            # pan duan shi fou cun zai wen jian
            # print(line[0:position])
            templabels.append(int(line[position + 1:-1]))

        self.samplename = temp  # 文件名
        self.label = templabels
        print(len(self.samplename))
        print(len(self.label))
        self.shuffle=shuffle
        self.originalindexes=np.arange(len(self.samplename))
        self.batch_size=batch_size
        self.data_dir=data_dir
        self.on_epoch_end()



    def on_epoch_end(self):#所有数据的索引为训练数据包含的患者ID的所有clip的片段号
        """每个epoch之后更新索引"""
        # self.indexes = np.arange(sum(self.clipsnumberofeachpatient))
        self.indexes=self.originalindexes
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def _generate_data2(self, list_IDs_temp):  # 包括输入x和输出y
        """生成每一批次的图像
        :param list_IDs_temp: 批次数据索引列表
        :return: 一个批次的图像"""
        # 初始化
        scaler=MinMaxScaler()
        traindata = []
        trainlabel = []

        # 生成数据
        for i, index1 in enumerate(list_IDs_temp):

            # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            # print(index1)
            # print(self.samplename[index1])
            tempnpysample1=cv2.imread(os.path.join(self.data_dir,self.samplename[index1]))
            tempnpysample1 = cv2.resize(tempnpysample1,(256,256))



            templabel=self.label[index1]

            trainlabel.append(templabel)
            traindata.append(tempnpysample1)
                    #  存储一个批次

        traindata = np.array(traindata)
        trainlabel=np.array(trainlabel)
        # traindata = scaler.transform(traindata)
        # traindata1 = scaler.fit_transform(traindata1)
        # trainlabel=trainlabel.reshape(trainlabel.shape[0])
        # trainlabel = to_categorical(trainlabel,2)
        return traindata,trainlabel





    def __getitem__(self, index):
        """生成每一批次训练数据
        :param index: 批次索引
        :return: 训练图像和标签
        :额外说明：在此函数中使用real_batch_size处理新版的数据（npy文件版本）
        """
        # 生成批次索引
        indexes=self.indexes[index * self.batch_size:(index+1) * self.batch_size]
        # 索引列表
        list_IDs_temp =[k for k in indexes]
        # 生成数据
        X,y = self._generate_data2(list_IDs_temp)
        return X,y



    def __len__(self):
        """每个epoch下的批次数量
        """
        return int(np.floor(len(self.samplename) / self.batch_size))


datagenerator1 = DataGenerator(data_dir='/home/zhl/copydata',patient_list_txtfile='train2.txt',batch_size=32,shuffle=True)
[a,b]=datagenerator1.__getitem__(0)
print("sssssssssssssssssssssssssssssssssssssssssssssssssssssss")
print(a.shape)
# now=next(datagenerator1)
# print(now[0][0].shape,now[0][0].shape)
# now=next(datagenerator1)
# print(now[0][0].shape,now[0][0].shape)
datagenerator2 = DataGenerator(data_dir='/home/zhl/copydata',patient_list_txtfile='test2.txt',batch_size=16,shuffle=True)
[a,b]=datagenerator2.__getitem__(0)
# print("==============================================")
# print(len(datagenerator2.radiomics))
print(a.shape)


def check_print():
    base_model=InceptionResNetV2(include_top=False,weights=None,input_shape=(256, 256, 3),backend=keras.backend,layers=keras.layers,models=keras.models,utils=keras.utils)
    # base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(256, 256, 3), backend=keras.backend,
    #                       layers=keras.layers, models=keras.models, utils=keras.utils)


    # base_model = MobileNetV2(include_top=False, weights=None, input_shape=(256, 256, 9))
    # base_model = MobileNet(include_top=False,weights=None,input_shape=(93, 151, 243))
    # base_model = DenseNet121(include_top=False, weights=None, input_shape=(256, 256, 3))
    # base_model = VGG16(include_top=False,weights=None,input_shape=(220, 220, 55))
    # base_model = InceptionV3(include_top=False, weights=None, input_shape=(256, 256, 3), backend=keras.backend,
    #                                layers=keras.layers, models=keras.models, utils=keras.utils)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2048,activation='relu')(x)
    # predictions = Dense(2, activation='softmax')(x)
    predictions = Dense(3, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.summary()
    # model.compile(optimizer=Adam(lr=0.00005, beta_1=0.9,beta_2=0.99,epsilon=1e-08,decay=1e-6),
    #                       loss='binary_crossentropy',
    #                       metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=0.00001,decay=1e-6),
    #                       loss='binary_crossentropy',
    #                       metrics=['accuracy'])
    # opti=optimizers.SGD(lr=0.01,momentum=0.95,decay=1e-6)
    model.compile(optimizer='sgd',
                          loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])
    # model.compile(optimizer=keras.optimizers.SGD(lr=0.001,momentum=0.9,decay=1e-6,nesterov=False),loss='binary_crossentropy',metrics=['accuracy'])
    return model



model = check_print()

# compile the model using mean absolute percentage error as our loss,
# implying that we seek to minimize the absolute percentage difference
# between our price *predictions* and the *actual prices*
# opt = Adam(lr=0.00005, beta_1=0.9,beta_2=0.99,epsilon=1e-08,decay=1e-6)
# opt=SGD(lr=0.01)
# opt = Adam(lr=0.01,decay=1e-6)
# model.compile(optimizer='SGD',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# model1.summary()
# train the model
print("[INFO] training model...")
#callbacks=[EarlyStopping(monitor='val_acc', patience=30, verbose=2, mode='max'),ModelCheckpoint(filepath='/home/zhl/mobilenetv2_101_1.h5',monitor='val_acc', verbose=1, save_best_only=True, mode='max')]
trained=model.fit_generator(generator=datagenerator1,validation_data=datagenerator2,epochs=80,class_weight={0:0.05,1:35.0,2:0.05},callbacks=[ModelCheckpoint(filepath='/home/zhl/inceptionresnet_2.h5',monitor='val_acc', verbose=1, save_best_only=True, mode='max')])
# print(trained.history['val_loss'])
# model.fit(
# 	[trainAttrX, trainImagesX], trainY,
# 	validation_data=([testAttrX, testImagesX], testY),
# 	epochs=100, batch_size=16,shuffle=True)

# model.fit(
# 	[trainAttrX, trainImagesX], trainY,
# 	validation_data=([testAttrX, testImagesX], testY),
# 	epochs=80, batch_size=16,class_weight={0:0.07,1:0.7,2:1.0},shuffle=True,callbacks=[EarlyStopping(monitor='roc_auc', patience=20, verbose=2, mode='max')])
# model.save('resnet_1.h5')
# make predictions on the testing data
print("[INFO] predicting...")
# preds = model.predict([testAttrX, testImagesX])
# print(preds.shape)
# np.savetxt('preds_4_resnet101.txt',preds)
