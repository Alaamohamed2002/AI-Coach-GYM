import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split
#from tensorflow.keras import layers, models
#from tensorflow.keras.utils import plot_model

# Define the video path
#video_path = "C:/Users/Aboa6/AppData/Local/project/projectGYM/Data/deadlift Correct/deadlift_1.MP4"
video_path = 'C:/Users/Aboa6/Desktop/Test/test.MP4'
model_save_path = 'The saved model/model.keras'
model = tf.keras.models.load_model(model_save_path)
# Load the trained model
# Replace this with your model loading code
# model = load_model()

# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Initialize MediaPipe Pose Detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame with MediaPipe Pose Detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)
    predicted_class_name = "Exercise False"
    # Check if pose landmarks are detected
    if result.pose_landmarks:
        # Extract pose landmarks
        landmarks = [[lm.x, lm.y] for lm in result.pose_landmarks.landmark]

        # Preprocess landmarks (reshape, convert to numpy array, etc.)
        # Example:
        landmarks_array = np.array(landmarks, dtype=np.float32)
        
        # Ensure landmarks_array has 33 landmarks
        num_landmarks = landmarks_array.shape[0]
        if num_landmarks < 33:
            # Pad with zeros to have 33 landmarks
            landmarks_array = np.pad(landmarks_array, ((0, 33 - num_landmarks), (0, 0)), mode='constant', constant_values=0)
        elif num_landmarks > 33:
            # Truncate to have 33 landmarks
            landmarks_array = landmarks_array[:33, :]
        
        landmarks_array = landmarks_array[np.newaxis, :, :]  # Add batch dimension

        # Set a threshold confidence level for model predictions
        confidence_threshold = 0.9  # A very high value

        # Use the model to make predictions
        predictions = model.predict(landmarks_array)

        # Check if the maximum confidence score is above the threshold
        max_confidence = np.max(predictions)
        if max_confidence >= confidence_threshold:
            # Get the index of the predicted class
            predicted_class_index = np.argmax(predictions)
     
            # Map the predicted class index to the exercise name
            if predicted_class_index == 1:
                predicted_class_name = "chest fly machine Correct"
            elif predicted_class_index == 3:
                predicted_class_name = "deadlift Correct"
            elif predicted_class_index == 5:
                predicted_class_name = "Trikasana Correct"
       # else:
            # If confidence is below threshold, return "Try Again"
            #predicted_class_name = "Exercise False"

        # Draw the predicted class name on the frame
        cv2.putText(frame, predicted_class_name, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("output", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
#accuracy = correct_predictions / total_frames
#print("Accuracy:", accuracy)
