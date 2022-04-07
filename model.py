# %%
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from itertools import product

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

DATA_PATH = os.path.join('Data') 
actions = np.array(os.listdir(os.path.join(DATA_PATH)))
no_sequences = 30
sequence_length = 30

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action, sequence in product(actions,range(no_sequences)):
    window = []
    for frame_num in range(sequence_length):
        res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
        window.append(res)
    sequences.append(window)
    labels.append(label_map[action])


X = np.array(sequences)
Y = to_categorical(labels).astype(int)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=34, stratify=Y)