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

dataset_path = 'C:/Users/Aboa6/AppData/Local/project/projectGYM/Data'
model_save_path = 'The saved model/model.keras'
NUM_CLASSES = 6
def load_dataset(dataset_path):
    x = []
    y = []
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Loop through dataset directory
    for exercise_folder in os.listdir(dataset_path):
        exercise_label = exercise_folder
        exercise_folder_path = os.path.join(dataset_path, exercise_folder)  #/content/Data1/crunches

        # Loop through video files in exercise folder
        for video_file in os.listdir(exercise_folder_path):
            video_path = os.path.join(exercise_folder_path, video_file)
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break
                # Process image using MediaPipe Pose Detection
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = pose.process(image_rgb)
                if result.pose_landmarks:
                    # Extract pose landmarks
                    landmarks = [[lm.x, lm.y] for lm in result.pose_landmarks.landmark]
                    x.append(landmarks)
                    y.append(exercise_label)
            cap.release()
    return np.array(x), np.array(y)
x, y = load_dataset(dataset_path)
print(f"Type of X: {type(x)}")
print(f"Type of y: {type(y)}")
m = 0
for i in y:
  if (i == 'chest fly machine Correct'):
    m+=1
print(m)

print(y)
for i in range(y.size):
  if y[i] == 'chest fly machine Wrong':
    y[i] = 0
  elif y[i] == 'chest fly machine Correct':
    y[i] = 1
  elif y[i] == 'deadlift Wrong':
    y[i] = 2
  elif y[i] == 'deadlift Correct':
    y[i] = 3
  elif y[i] == 'Trikasana Wrong':
    y[i] = 4
  elif y[i] == 'Trikasana Correct':
    y[i] = 5
  else:
    y[i] = -1
    
    print(y)
    if x.dtype != np.float32:
        x = x.astype(np.float32)
if y.dtype != np.float32:
    y = y.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    #building the model
    model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(256, kernel_size=3, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
    model_save_path, verbose=1, save_weights_only=False)
    
    model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
    
    #train the model
    history = model.fit(
    X_train,
    y_train,
    epochs=300,
    batch_size=64,
    validation_split=0.2,
    callbacks=[cp_callback]
    )
    model.summary()
    loss, accuracy = model.evaluate(X_test, y_test)
model = tf.keras.models.load_model(model_save_path)
print(model.input_shape)
#video_path = 'C:/Users/Aboa6/Desktop/Test'
video_path = "C:/Users/Aboa6/AppData/Local/project/projectGYM/Data"
cap = cv2.VideoCapture(video_path)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame with MediaPipe Pose Detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)

    # Check if pose landmarks are detected
    if result.pose_landmarks:
        # Extract pose landmarks
        landmarks = [[lm.x, lm.y] for lm in result.pose_landmarks.landmark]

        # Preprocess landmarks (reshape, convert to numpy array, etc.)
        # Example:
        landmarks_array = np.array(landmarks, dtype=np.float32)
        landmarks_array = landmarks_array[np.newaxis, ...]  # Add batch dimension

        # Use the model to make predictions
        predictions = model.predict(landmarks_array)

        # Example: Print the predicted class
        if (np.argmax(predictions) == 0):
          print("Predicted Class: chest fly machine Wrong")
        elif (np.argmax(predictions) == 1):
          print("Predicted Class: chest fly machine Correct")
        elif (np.argmax(predictions) == 2):
          print("Predicted Class: deadlift Wrong")
        elif (np.argmax(predictions) == 3):
          print("Predicted Class: deadlift Correct")
        elif (np.argmax(predictions) == 4):
          print("Predicted Class: Trikasana Wrong")
        elif (np.argmax(predictions) == 5):
          print("Predicted Class: Trikasana Correct")
        else:
          print("nothing")

    # Display the frame
    cv2.imshow("output",frame)
    

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
