import librosa
import numpy as np

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