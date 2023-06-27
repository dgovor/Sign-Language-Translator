import mediapipe as mp
import cv2
import numpy as np

def draw_landmarks(image, results):
    """
    Draw the landmarks on the image.

    Args:
        image (numpy.ndarray): The input image.
        results: The landmarks detected by Mediapipe.

    Returns:
        None
    """
    # Draw landmarks for left hand
    mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
    # Draw landmarks for right hand
    mp.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)

def image_process(image, model):
    """
    Process the image and obtain sign landmarks.

    Args:
        image (numpy.ndarray): The input image.
        model: The Mediapipe holistic object.

    Returns:
        results: The processed results containing sign landmarks.
    """
    # Set the image to read-only mode
    image.flags.writeable = False
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process the image using the model
    results = model.process(image)
    # Set the image back to writeable mode
    image.flags.writeable = True
    # Convert the image back from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return results

def keypoint_extraction(results):
    """
    Extract the keypoints from the sign landmarks.

    Args:
        results: The processed results containing sign landmarks.

    Returns:
        keypoints (numpy.ndarray): The extracted keypoints.
    """
    # Extract the keypoints for the left hand if present, otherwise set to zeros
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    # Extract the keypoints for the right hand if present, otherwise set to zeros
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    # Concatenate the keypoints for both hands
    keypoints = np.concatenate([lh, rh])
    return keypoints
