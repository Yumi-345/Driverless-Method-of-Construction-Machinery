__Author__ = "xq"


from sklearn.model_selection import train_test_split
from keras.layers import Input, Conv2D, Dense, BatchNormalization, MaxPool2D, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import tanh
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.optimizers import RMSprop
import pandas as pd
import numpy as np
import cv2, os

DATA_PATH = "./Data20191209/Data.csv"
MODELSPATH = "./Data20191209/Model/Model.h5"
BATCH_SIZE = 2
EPOCHES = 20
IMG_WIDTH = 480
IMG_HEIGHT = 640

def get_data(data_path):
    data = pd.read_csv(data_path)
    x, y = data[["IMGPath"]].values, data[["Steering"]].values
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)
    return x_train, x_test, y_train, y_test

def get_model():
    input = Input(shape=(480, 640, 3))
    x = MaxPool2D((2,2), 2)(input) # 320x240x3

    x = Conv2D(8, (5,5), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPool2D((2,2), 2)(x) # 160x120x8

    x = Conv2D(16, (5,5), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPool2D((2,2), 2)(x) # 80x60x16

    x = Conv2D(32, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(32, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPool2D((2,2), 2)(x) # 40x30x32

    x = Conv2D(64, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(64, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPool2D((2,2), 2)(x) # 20x15x64

    x = Conv2D(64, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(64, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPool2D((2,2), 2)(x) # 10x7x64

    x = Flatten()(x)
    x = Dropout(0.3)(x)

    x = Dense(1000)(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)

    x = Dense(10)(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)

    out = Dense(1, activation="tanh")(x)


    model = Model(inputs=input, outputs=out)
    model.compile(optimizer=RMSprop(0.0001), loss="mse", metrics=["accuracy"])
    model.summary()
    return model

def batch_generator(x_train, y_train, is_train):
    images = np.empty([BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, 3])
    labels = np.empty([BATCH_SIZE])
    while True:
        i = 0
        for index in np.random.permutation(x_train.shape[0]):
            rgb_img_path = "./Data20191209" + x_train[index][0]
            label = y_train[index][0]
            rgb_img = cv2.imread(rgb_img_path)

            img = (rgb_img/255.0).astype(np.float32)
            images[i] = img
            labels[i] =label

            i += 1
            if i == BATCH_SIZE:
                break
        yield images, labels

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = get_data(DATA_PATH)
    model = get_model()
    callbacks = [
        # EarlyStopping(monitor="val_dice_loss",
        #                    patience=5,
        #                    verbose=1,
        #                    min_delta=1e-4,
        #                    mode="max"),
        ModelCheckpoint(MODELSPATH,
                     monitor="val_loss",
                     verbose=0,
                     save_best_only=True,
                     mode="auto")]
    model.fit_generator(batch_generator(x_train, y_train, True),
                        steps_per_epoch=len(x_train)//BATCH_SIZE,
                        epochs=EPOCHES,
                        # max_queue_size=1,
                        validation_data=batch_generator(x_test, y_test, False),
                        validation_steps=len(x_test),
                        callbacks=callbacks,
                        verbose=1)

