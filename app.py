import streamlit as st
from main import feature_extract
import time
import joblib

model=joblib.load("model.pkl")
scaler=joblib.load("scaler.pkl")

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


st.title("Voice emotion ✨ detection")

st.subheader("Upload Audio")
file_path=st.file_uploader(
    "Upload your audio",
    type=['wav', 'mp3', 'ogg']
)
if file_path is not None:
    emotion=prediction(file_path).capitalize()
    with st.spinner():
        time.sleep(2)
        st.success(f"Detected Emotion: {emotion}")
else:
    st.warning("Please upload an audio file")