# %%
import cv2
import numpy as np
import os
import time 
import mediapipe as mp
from itertools import product
from my_functions import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

DATA_PATH = os.path.join('Data') 

#actions = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])

actions = np.array(['c', 'd'])

no_sequences = 30
sequence_length = 10

for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

cap = cv2.VideoCapture(0)

with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action, sequence, frame_num in product(actions, range(no_sequences), range(sequence_length)):

        ret, frame = cap.read()

        image, results = mediapipe_detection(frame, holistic)

        draw_landmarks(image, results)

        cv2.putText(image, 'Collecting frames for the letter "{}". Video Number {}'.format(action, sequence), (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        if frame_num == 0: 
            cv2.putText(image, 'PAUSE', (15,50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
            cv2.putText(image, 'Press any key to continue', (15,80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
            cv2.imshow('OpenCV feed', image)
            cv2.waitKey(0)
        else: 
            cv2.imshow('OpenCV feed', image)
        
        keypoints = extract_keypoints(results)
        npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
        np.save(npy_path, keypoints)

        cv2.waitKey(1)
        if cv2.getWindowProperty('OpenCV feed',cv2.WND_PROP_VISIBLE) < 1:
            break
                    
    cap.release()
    cv2.destroyAllWindows()
