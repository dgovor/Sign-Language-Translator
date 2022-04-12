# %%
import numpy as np
import os
import mediapipe as mp
import cv2
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from itertools import product
from sklearn import metrics
from my_functions import *

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv2D
from tensorflow.keras.callbacks import TensorBoard

DATA_PATH = os.path.join('Data') 
actions = np.array(os.listdir(os.path.join(DATA_PATH)))
no_sequences = 30
sequence_length = 10

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action, sequence in product(actions, range(no_sequences)):
    window = []
    for frame_num in range(sequence_length):
        res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
        window.append(res)
    sequences.append(window)
    labels.append(label_map[action])


X = np.array(sequences)
Y = to_categorical(labels).astype(int)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=34, stratify=Y)

# %%
tb_callback = TensorBoard(log_dir=os.path.join('Logs'))

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(10,126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, Y_train, epochs=100, callbacks=[tb_callback])
