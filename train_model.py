import sys
import os
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

df = pd.read_csv('fer2013.csv')

X_full = df["pixels"].str.split(" ", expand=True).to_numpy(dtype="float32")
y_full = df["emotion"].to_numpy()

num_features = 64
num_labels = 7
batch_size = 64
epochs = 30
width, height = 48, 48

y_full = to_categorical(y_full, num_classes=num_labels)

# cannot produce
# normalizing data between oand 1
X_full /= 255.0

X_full = X_full.reshape(X_full.shape[0], 48, 48, 1)

print("Processed data")

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
model.add(Dense(num_labels, activation='softmax'))

# Compliling the model
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])

# Training the model
model.fit(X_full, y_full,
          batch_size=batch_size,
          epochs=100,
          verbose=1,
          shuffle=True)

# Saving the  model to  use it later on
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
