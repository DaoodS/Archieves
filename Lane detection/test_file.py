import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import cv2

data = pd.read_csv("C:\\Users\\Daood-PC\\Desktop\\finale.csv", encoding  = 'utf-8')
y1 = data['steering']
y2 = data['throttle']
y3 = data['brake']
x1 = data['center']
x = np.zeros(shape=(1,160,320,3))
n = random.randint(1, 300)

print("Random number generated = ",n)

img = cv2.imread(x1[n])
x = np.concatenate((x,np.array(img).reshape(1,160,320,3)))
y1_final = y1[n]
y2_final = y2[n]
y3_final = y3[n]


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
model1.load_weights('MODEL1.h5')

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
model2.load_weights('model2.h5')

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
model3.load_weights('model3.h5')

predicted_y1 = model1.predict(x, verbose = 0)
predicted_y2 = model2.predict(x, verbose = 0)
predicted_y3 = model3.predict(x, verbose = 0)

plt.imshow(img)
plt.show()
print("Labelled steering angle =", y1_final, ": Calculated steering angle = ", predicted_y1[1])
print("Labelled throttle =", y2_final, ": Calculated throttle = ", predicted_y2[1])
print("Labelled brake =", y3_final, ": Calculated brake = ", predicted_y3[1])