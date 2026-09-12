import librosa
import numpy as np
from pathlib import Path

def extract_mfcc(file_path, n_mfcc=40, max_len=174):
    """Return a fixed-size MFCC matrix for an audio file."""
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

def load_data(data_dir, emotions, n_mfcc=40, max_len=174):
    X, y = [], []
    for idx, emotion in enumerate(emotions):
        emotion_dir = Path(data_dir) / emotion
        if not emotion_dir.is_dir():
            raise FileNotFoundError(f"Emotion folder not found: {emotion_dir}")
        for file in emotion_dir.iterdir():
            if file.suffix.lower() == '.wav':
                mfcc = extract_mfcc(file, n_mfcc, max_len)
                X.append(mfcc)
                y.append(idx)
    return np.array(X), np.array(y)
