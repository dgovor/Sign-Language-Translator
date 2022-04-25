# Sign-Language-Translator
**Work in Progress** (The code will get the comments in the upcoming days)

Neural Network created using Sequential architecture and combination of LSTM and Dense layers in order to translate American Sign Language (ASL) into text.

<p align="center"> <img src="img/1.gif" alt="drawing" width="450"/> </p>


## Description

This project provides an opportunity for people to train their own Neural Network by recording their own dataset of ASL signs in an intuitive and simple manner.
The whole project can be split into three main parts:
1. Data collection;
2. Model training;
3. Real time predictions.

## Data Collection

In order for a user to collect data and create their own dataset, the [data_collection.py](https://github.com/dgovor/Sign-Language-Translator/blob/main/data_collection.py) is used. The script is organized in a way that it would be easy to configure your own preferences and options, such as the signs the user would like to add to their dataset, the number of sequences for each sign, the number of frames for each sequence, and the path where the user would like to store the dataset. Onces these parameters were set and the script is running, the user can start recording the data. <ins>It is recommended that the user record a substantial number of sequences changing the position of their hands. This way the user can ensure data diversity which helps to obtain a generalized model.</ins>

<p align="center"> <img src="img/2.gif" alt="drawing" width="450"/> </p>

[MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic) pipeline was used to record the data from the user's hands. Using [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic) instead of [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands) opens doors to future extensions and possibilities of this script. The pipeline processes each frame sent through it and results in the pose, face, left hand, and right hand components neatly stored in a variable. Each of the components can be represented by landmarks (these components' coordinates). In this case, only the hands' components' landmarks are being extracted resulting in overall 126 data entries (21 landmarks per hand with _x_, _y_, _z_ coordinates per landmark).



## Model Training

Coming soon.

## Real Time Predictions

Coming soon.
