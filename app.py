import streamlit as st
from main import prediction
import time
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