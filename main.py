from utils import feature_extract # Audio loading and feature extraction
import numpy as np  # Numerical array
import os           # File and folder handling
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import streamlit as st
import joblib


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

model = joblib.dump(model,"model.pkl")
scaler = joblib.dump(scaler,"scaler.pkl")






