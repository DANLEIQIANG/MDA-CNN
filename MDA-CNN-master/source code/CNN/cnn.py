# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import argparse
#import data_helpers as dh
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
#from tensorflow.contrib import learn
import matplotlib.pyplot as plt
import sklearn.preprocessing as prep
from sklearn.metrics import average_precision_score
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D



train_data = pd.read_csv('/Users/qiangdanlei/Desktop/graduate thesis/MDA-CNN-master/data/CNN/disease_miRNA.csv')
train_data=pd.DataFrame(train_data)
#train_data=train_data.drop([0])
#train_data=train_data.drop([0], axis = 1)
train_data=train_data.T
train_data=np.array(train_data)
#train_data = np.array(train_data)ç
print('done reading data')
train_label = pd.read_csv('/Users/qiangdanlei/Desktop/graduate thesis/MDA-CNN-master/data/CNN/label1.csv', header=None)
print("done reading label")
train_label = np.array(train_label[0]).tolist()
#print(train_data)
preprocessor = prep.StandardScaler().fit(train_data)
train_data= preprocessor.transform(train_data)

X_train, X_test, y_train, y_test = train_test_split(train_data,train_label,test_size=0.2, random_state=10)

print(len(X_test))
X_train=X_train.reshape(318,1024,1).tolist()

X_test=X_test.reshape(80,1024,1).tolist()


model = Sequential()

#第一个 卷积层 的卷积核的数目是32 ，卷积核的大小是3*3，stride没写，默认应该是1*1
#对于stride=1*1,并且padding ='same',这种情况卷积后的图像shape与卷积前相同，本层后shape还是32*32
model.add(Conv1D(kernel_size=4,
                 filters=4,
                 padding='valid',
                 strides=4,
                 input_shape=(1024,1)))
model.add(Activation('relu'))
#keras Pool层有个奇怪的地方，stride,默认是(2*2),padding 默认是valid，在写代码是这些参数还是最好都加上
model.add(  MaxPooling1D(pool_size=4,strides=4,padding='same')  )

model.add(Flatten())

model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('softmax'))

model.summary()

# initiate RMSprop optimizer
opt = keras.optimizers.Adam(lr=0.001, decay=1e-6)

# train the model using RMSprop
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit([X_train], [y_train], batch_size=256, epochs=10,
                    validation_split=0.1)
print([y_train])
# evaluate
res = model.evaluate([X_test], y_test)








'''
model = keras.Sequential()
# 卷积层 1
model.add(keras.layers.Conv1D(input_shape=(1024,1),
                        filters=4, kernel_size=4,  padding='valid',  activation='relu'))


model.add(keras.layers.MaxPool1D(pool_size=4,strides=4, padding="valid"))


# 全连接层
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(50, activation='relu'))

# 分类层
model.add(keras.layers.Dense(1, activation='softmax'))

# 3. 模型配置
model.compile(optimizer=keras.optimizers.Adam(),
              loss='mean_squared_error',
              metrics=['accuracy'])

model.summary()
# 4. 模型训练
model.fit(X_train, y_train, batch_size=64, epochs=100,
                    validation_split=0.1)

#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
#plt.legend(['training', 'valivation'], loc='upper left')
#plt.show()

# 5. 模型评估
a=model.predict(X_test)
#print(a)
#print(y_test)
res = model.evaluate(X_test, y_test)
'''
