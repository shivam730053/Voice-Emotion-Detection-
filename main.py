import librosa  # Audio loading and feature extraction
import numpy as np  # Numerical array
import os           # File and folder handling
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import streamlit as st


def feature_extract(file_path):
    """
    Load an audio file and extract MFCC features.

    Steps:
    1. Load 3 seconds of audio, skipping first 0.5s (often silence)
    2. Extract 40 MFCC coefficients
    3. Average across time to get fixed-size vector
    """

    # MFCC Shpae = (40, time_frames)
    audio, sr=librosa.load(file_path, duration=3, offset=0.5)
    mfcc=librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    # Convert variable-length audio into fixed-size feature vector 
    return np.mean(mfcc.T, axis=0)

emotions_map={
    "01": "neutral",
    "02": "calm",
    "03": "happy", 
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

X=[]  # Feature vectors
y=[]  # Emotion labels
dataset_path='data'

for actor in os.listdir(dataset_path):
    actor_path=os.path.join(dataset_path, actor)
    # Skip non-folder items
    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):
        # Process only wave files
        if not file.endswith(".wav"):
            continue

        modality=file.split("-")[1] # For speech only
        
        # Keeps only speech(01), ignore song(02)
        if modality  != "01":
            continue
        
        # Extract emotion code from file name
        emotion_code=file.split("-")[2]
        emotion=emotions_map[emotion_code] 
        # Full path to audio file
        file_path=os.path.join(actor_path, file)
        # Extract mfcc features
        features=feature_extract(file_path)

        # Store data
        X.append(features)
        y.append(emotion)

# Convert list into numpy array for sklearn
X=np.array(X)
y=np.array(y)

# stratify = y (Keeps class distribution balanced)
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# SVM requires normalize features
scaler=StandardScaler()
# Fit only on training data
X_train=scaler.fit_transform(X_train)
# Apply same scaling to test data
X_test=scaler.transform(X_test)

model=SVC(kernel='rbf', C=10, gamma='scale', probability=True)
model.fit(X_train, y_train)

y_pred=model.predict(X_test)
# Shows precesion, recall, f1 score on every emotion
print(classification_report(y_test, y_pred))

def prediction(file_path):
    """
    Predict emotion from a new audio file.
    """

    # Extract features
    features=feature_extract(file_path)
    # Reshape the match sklearn inputs: (sample, features)
    features=features.reshape(1, -1)
    # Applying same transformation as training data
    features=scaler.transform(features)  # Required
    # Predict emotion label
    return model.predict(features)[0]






