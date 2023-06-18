# %%

# Import necessary libraries
import numpy as np
import os
import mediapipe as mp
import cv2
from my_functions import *
from tensorflow.keras.models import load_model

# Set the path to the data directory
PATH = os.path.join('data')

# Create an array of action labels by listing the contents of the data directory
actions = np.array(os.listdir(PATH))

# Load the trained model
model = load_model('my_model')

# Initialize the sentence and keypoints lists
sentence, keypoints = [' '], []

# Access the camera and check if the camera is opened successfully
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

# Create a holistic object for sign prediction
with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
    # Run the loop while the camera is open
    while cap.isOpened():
        # Read a frame from the camera
        _, image = cap.read()
        # Process the image and obtain sign landmarks using image_process function from my_functions.py
        results = image_process(image, holistic)
        # Draw the sign landmarks on the image using draw_landmarks function from my_functions.py
        draw_landmarks(image, results)
        # Extract keypoints from the pose landmarks using keypoint_extraction function from my_functions.py
        keypoints.append(keypoint_extraction(results))

        # Check if 10 frames have been accumulated
        if len(keypoints) == 10:
            # Convert keypoints list to a numpy array
            keypoints = np.array(keypoints)
            # Make a prediction on the keypoints using the loaded model
            prediction = model.predict(keypoints[np.newaxis, :, :])
            # Clear the keypoints list for the next set of frames
            keypoints = []

            # Check if the maximum prediction value is above 0.9
            if np.amax(prediction) > 0.9:
                # Check if the predicted sign is different from the previous sign in the sentence
                if sentence[-1] != actions[np.argmax(prediction)]:
                    # Append the predicted sign to the sentence list
                    sentence.append(actions[np.argmax(prediction)])

        # Limit the sentence length to 7 elements to make sure it fits on the screen
        if len(sentence) > 7:
            sentence = sentence[-7:]

        # Reset the sentence if the spacebar is pressed
        if keyboard.is_pressed(' '):
            sentence = [' ']

        # Calculate the size of the text to be displayed and the X coordinate for centering the text on the image
        textsize = cv2.getTextSize(' '.join(sentence), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_X_coord = (image.shape[1] - textsize[0]) // 2

        # Draw the sentence on the image
        cv2.putText(image, ' '.join(sentence), (text_X_coord, 470), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the image on the display
        cv2.imshow('Camera', image)
        
        cv2.waitKey(1)
        if cv2.getWindowProperty('Camera',cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
