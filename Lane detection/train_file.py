import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
import cv2

data = pd.read_csv("C:\\Users\\Daood-PC\\Desktop\\finale.csv", encoding  = 'utf-8')
y1 = data['steering']
y2 = data['throttle']
y3 = data['brake']

#temp = cv2.imread("C:\\Users\\Daood-PC\\Desktop\\DT\\IMG\\center_2018_03_16_19_18_28_795.jpg")
#print(temp)
x = np.zeros(shape=(1,160,320,3))
for index,row in data.iterrows():
    img = cv2.imread(row['center'])
    # print(x.shape)
    #print(np.array(img).shape)
    # img.reshape(1,160,320,3)
    #if(np.array(img).shape == (1, 160, 320, 3)):
    #if img is not None:
    # print(img,index)
    x = np.concatenate((x,np.array(img).reshape(1,160,320,3)))
x = x[1:]

model1 = Sequential()
model1.add(Conv2D(8, (9, 9),strides = 4, input_shape = (160,320,3)))
model1.add(Activation('relu'))
model1.add(Conv2D(16, (5, 5),strides = 2))
model1.add(Activation('relu'))
model1.add(Conv2D(32, (5, 5),strides = 2))
model1.add(Activation('relu'))
model1.add(Conv2D(64, (3, 3),strides = 2))
model1.add(Flatten())
model1.add(Dropout(0.2))
model1.add(Activation('relu'))
model1.add(Dense(1024))
model1.add(Dropout(0.5))
model1.add(Activation('relu'))
model1.add(Dense(1))

model1.compile(loss='mean_absolute_error',
              optimizer='adam',
              metrics=['mean_absolute_error'])
batch_size = 16
model1.fit(x,y1,verbose = 2,epochs = 20)
model1.save_weights('MODEL1.h5')

model2 = Sequential()
model2.add(Conv2D(8, (9, 9),strides = 4, input_shape = (160,320,3)))
model2.add(Activation('relu'))
model2.add(Conv2D(16, (5, 5),strides = 2))
model2.add(Activation('relu'))
model2.add(Conv2D(32, (5, 5),strides = 2))
model2.add(Activation('relu'))
model2.add(Conv2D(64, (3, 3),strides = 2))
model2.add(Flatten())
model2.add(Dropout(0.2))
model2.add(Activation('relu'))
model2.add(Dense(1024))
model2.add(Dropout(0.5))
model2.add(Activation('relu'))
model2.add(Dense(1))

model2.compile(loss='mean_absolute_error',
              optimizer='adam',
              metrics=['mean_absolute_error'])
batch_size = 16
model2.fit(x,y2,verbose = 2,epochs = 20)
model2.save_weights('model2.h5')

model3 = Sequential()
model3.add(Conv2D(8, (9, 9),strides = 4, input_shape = (160,320,3)))
model3.add(Activation('relu'))
model3.add(Conv2D(16, (5, 5),strides = 2))
model3.add(Activation('relu'))
model3.add(Conv2D(32, (5, 5),strides = 2))
model3.add(Activation('relu'))
model3.add(Conv2D(64, (3, 3),strides = 2))
model3.add(Flatten())
model3.add(Dropout(0.2))
model3.add(Activation('relu'))
model3.add(Dense(1024))
model3.add(Dropout(0.5))
model3.add(Activation('relu'))
model3.add(Dense(1))

model3.compile(loss='mean_absolute_error',
              optimizer='adam',
              metrics=['mean_absolute_error'])
batch_size = 16
model3.fit(x,y3,verbose = 2,epochs = 20)
model3.save_weights('model3.h5')

