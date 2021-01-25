import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


def preprocess_data():
    df = pd.read_csv('fer2013.csv')

    X_full = df["pixels"].str.split(" ", expand=True)
    y_full = df["emotion"]
    df = pd.concat([X_full, y_full], axis=1)

    train, test = train_test_split(df, test_size=0.1)

    X_train = train[train.columns.drop("emotion")].to_numpy(dtype="float32")
    y_train = train["emotion"].to_numpy()
    X_test = test[test.columns.drop("emotion")].to_numpy(dtype="float32")
    y_test = test["emotion"].to_numpy()

    y_train = to_categorical(y_train, num_classes=7)
    y_test = to_categorical(y_test, num_classes=7)

    # normalizing data between 0 and 1
    X_train /= 255.0
    X_test /= 255.0

    X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
    X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

    print("Finished preprocessing the data")

    return X_train, y_train, X_test, y_test


def compile_cnn_model():
    # designing the cnn
    # 1st convolution layer
    model = Sequential()

    model.add(Conv2D(64, (3, 3), activation='relu',
                     padding='same', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 2nd convolution layer
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu'))

    # 3rd convolution layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    # Compliling the model
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(),
                  metrics=['accuracy'])

    return model


def train_cnn_model(model, X_train, y_train, X_test, y_test):
    es = EarlyStopping(monitor="val_loss", mode="min",
                       patience=20, restore_best_weights=True)
    # Training the model
    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              batch_size=64,
              epochs=500,
              verbose=1,
              callbacks=[es],
              shuffle=True)
    train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Train accuracy: {train_accuracy} Test accuracy: {test_accuracy}")


def save_model(model):
    # Saving the  model to  use it later on
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")


def main():
    X_train, y_train, X_test, y_test = preprocess_data()
    model = compile_cnn_model()
    train_cnn_model(model, X_train, y_train, X_test, y_test)
    save_model(model)


if __name__ == "__main__":
    main()
