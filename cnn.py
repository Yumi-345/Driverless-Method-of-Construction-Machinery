from keras.layers import Input, UpSampling2D, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model
import numpy as np
import keras.backend as K
import cv2

videoCapture = cv2.VideoCapture(0)

input = Input(shape=(640,480,3))
x = MaxPooling2D(2)(input)

x = Conv2D(8, (3,3), padding='same', activation='relu')(x)
x = Conv2D(8, (3,3), padding='same', activation='relu')(x)
x = MaxPooling2D(strides=2)(x) #(160, 120, 8)

x = Conv2D(8, (3,3), padding='same', activation='relu')(x)
x = Conv2D(8, (3,3), padding='same', activation='relu')(x)
x = MaxPooling2D(strides=2)(x) #(80, 60, 8)

x = Conv2D(8, (3,3), padding='same', activation='relu')(x)
x = Conv2D(8, (3,3), padding='same', activation='relu')(x)
x = MaxPooling2D(strides=2)(x) #(40, 30, 8)

x = Flatten()(x)

x = Dense(512, activation='relu')(x)
x = Dense(32, activation='relu')(x)
output = Dense(2, activation='sigmoid')(x)
# x = UpSampling2D()(input)

model = Model(inputs=input, outputs=output)
model.compile(optimizer="sgd", loss="mse")

while True:
    success, frame = videoCapture.read()
    cv2.imshow('name', frame)
    frame = np.reshape(frame, (-1, 640, 480, 3))
    y = model.predict(frame)
    print(y[0][0], y[0][1])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# y = np.squeeze(y, (0,-1))
# print(y)


