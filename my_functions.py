import mediapipe as mp
import cv2
import numpy as np

def draw_landmarks(image, results):
    mp.solutions.drawing_utils.draw_landmarks(image, results.face_landmarks, mp.solutions.holistic.FACEMESH_CONTOURS,
                              mp.solutions.drawing_utils.DrawingSpec(color=(0,256,0), thickness=1, circle_radius=1), 
                              mp.solutions.drawing_utils.DrawingSpec(color=(0,256,0), thickness=1, circle_radius=1))
    mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS,
                              mp.solutions.drawing_utils.DrawingSpec(color=(0,256,0), thickness=2, circle_radius=2), 
                              mp.solutions.drawing_utils.DrawingSpec(color=(0,256,0), thickness=1, circle_radius=1))
    mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS,
                              mp.solutions.drawing_utils.DrawingSpec(color=(0,256,0), thickness=2, circle_radius=2), 
                              mp.solutions.drawing_utils.DrawingSpec(color=(0,256,0), thickness=1, circle_radius=1))
    mp.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS,
                              mp.solutions.drawing_utils.DrawingSpec(color=(0,256,0), thickness=2, circle_radius=2), 
                              mp.solutions.drawing_utils.DrawingSpec(color=(0,256,0), thickness=1, circle_radius=1))

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, face, lh, rh])