# Sign-Language-Translator

This project is aimed at developing a Neural Network using LSTM and Dense layers to translate any sign language into text. It provides a user-friendly way for individuals to train their own Neural Network model and enables real-time predictions as well as grammar correction of predicted sentences. 

### Key Features:
* User-friendly data collection process for creating custom sign language datasets.
* Training of a Neural Network model using LSTM and Dense layers.
* Real-time predictions of hand gestures based on hand landmarks.
* Integration of GingerIt library to perform grammar correction.
* Incorporation of MediaPipe Holistic pipeline for accurate hand tracking.

<p align="center"> <img src="img/1_1.gif" alt="drawing" width="450"/> </p>


## Description

This project provides an opportunity for people to train their own Neural Network by recording their own dataset of hand gestures in an intuitive and simple manner.
The whole project can be split into three main parts:
1. Data collection.
2. Model training.
3. Real time predictions.

## Data Collection

In order for a user to collect data and create their own dataset, the [data_collection.py](https://github.com/dgovor/Sign-Language-Translator/blob/main/data_collection.py) is used. The script is organized in a way that it would be easy to configure your own preferences and options, such as the signs the user would like to add to their dataset, the number of sequences for each sign, the number of frames for each sequence, and the path where the user would like to store the dataset. Once these parameters were set and the script is running, the user can start recording the data. <ins>It is recommended that the user records a substantial number of sequences changing the position of their hands. This way the user can ensure data diversity which helps to obtain a generalized model.</ins>

<p align="center"> <img src="img/1_2.gif" alt="drawing" width="450"/> </p>

[MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic) pipeline was used to record the data from the user's hands. Using [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic) instead of [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands) opens doors to future extensions and possibilities of this script. The pipeline processes each frame sent through it and results in the pose, face, left hand, and right hand components neatly stored in a variable. Each of the components can be represented by landmarks (these components' coordinates). In this case, only the hands' components' landmarks are being extracted resulting in overall 126 data entries (21 landmarks per hand with _x_, _y_, _z_ coordinates per landmark).

## Model Training

After the data has been collected and the dataset is complete, the user can proceed with the model training. In this step, the dataset is split into two subsets: 90% of the dataset is used for training and 10% for testing. The accuracy of testing using this 10% of the dataset will provide insight into the efficiency of the model.

For this particular project, the Neural Network is built using a Sequential model instance by passing three LSTM and two Densely-connected layers. The first four of these layers use the ReLU activation function with the last layer using the Softmax activation function. In the process of training, the Adam optimization algorithm is used to obtain optimal parameters for each layer.

Once the Neural Network is compiled, one can proceed with the model training and testing. During this step, the user can provide the model with the training subset, associated labels, and the number of epochs. Depending on the size of the provided subset and the number of epochs the training process can take up to a few minutes. Following the training, one can assess the model by performing predictions using the testing subset and evaluating the accuracy of these predictions.

## Real Time Predictions

In this step, the Neural Network is ready to apply everything it has learned to the real-world problem. [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic) pipeline processes every frame captured by a video camera and extracts hands' landmarks. Every new frame the script appends the landmarks to the previous ones until it reaches the length 10. Once 10 frames are processed and the corresponding landmarks are grouped together, the script converts the list with all the landmarks into an array and passes this array to the trained Neural Network so it can predict the sign of the user's hands. The prediction is then appended to the sentence list initialized earlier and the first word of the sentence is capitalized. Once the user finished recording the sentence they can press "Enter" to perform a grammar check and correction. If the user is not satisfied with the result they can press the "Spacebar" to reset the lists and start over.

## Conclusion

By combining advanced machine learning techniques and real-time hand tracking, Sign-Language-Translator empowers individuals to bridge the communication gap between sign language gestures and text, facilitating effective communication for the deaf and hearing-impaired.
