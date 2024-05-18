## Fitness Exercise Classifier Model (Version 2)
## Overview

This repository contains a deep learning model for classifying fitness exercises based on Frames extracted from videos. The model is built using TensorFlow and Opencv and Mediapipe and is capable of recognizing various exercises such as Lunges, Side Plank, and Upper Crunches.

<br>

## Installation

### Clone the repository

```bash

git clone https://github.com/computervisioneng/sign-language-detector-python.git

```
<br>

1. Navigate to the project directory:

2. Install dependencies
<br>

```bash

pip install -r requirements.txt

```

<br>

## Model Architecture

<br>
The model architecture is based on a convolutional neural network (CNN) implemented using TensorFlow's Keras API. It consists of multiple convolutional layers followed by fully connected layers, ending with a softmax activation function to output class probabilities.

<br>

## Dataset
<br>
The model was trained on a custom dataset consisting of videos of individuals performing different fitness exercises. These videos were annotated with pose landmarks using the MediaPipe Pose Detection framework.

<br>
<br>

## IMAGE COLLECTING:
 <br>

-to collect images: 
1- Collect Videos of people Performing the exercises you want
<br>
2- run 'DataCollection.py' to start extracting Frames of the videos

<br> 

## DATA PROCESSING:

- after collecting all the Videos and running DataCollection.py to extract frames ..
, Run the file 'DataPreprocessing.py' to extract landmarks and store them .

<br>
<br>

## TRAINING:

run the file ModelTrainer2.py to start training the model
<br>

## TESTING:

run the file AITrainer.py to start testing the model using videos

<br>

## EVALUATION

-run the file 'ModelTraining.py' to evalute model performance and calculate accuracy ,and to produce the confusion_matrix
<br>


## Files
<br>


DataCollection.py: this script is used to extract frames from videos and storing them is classes
<br>
DataPreprocessing.py :  this script is used to extract Landmarks from the Photos and storing them in Numpy array
<br>

AiTrainer.py: This script allows for real-time inference using the trained model on webcam or video input. It loads the trained model, processes input frames using MediaPipe Pose Detection, and outputs predictions for each frame.
<br>

ModelTraining2.py: This script is used for training the model using the provided dataset. It loads the dataset, preprocesses the data, defines and compiles the model architecture, trains the model, and saves the trained model to a specified location.
also This script is for evaluating the model's performance and generating analysis reports. It loads the trained model, evaluates it on a separate test dataset, calculates evaluation metrics such as accuracy, precision, recall, and F1-score, and generates a confusion matrix for visualizing the model's performance across different classes.
<br>

requirements.txt: This file contains a list of dependencies required to run the scripts. You can install these dependencies using
```bash
 pip install -r requirements.txt.

 ```
<br>



