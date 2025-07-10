import numpy as np
from tensorflow.keras.models import load_model
from features import extract_mfcc

emotions = ['happy', 'angry', 'sad']  # Same order as training

def predict_emotion(file_path):
    mfcc = extract_mfcc(file_path)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]  # Add batch and channel dims
    model = load_model('emotion_cnn.h5')
    pred = model.predict(mfcc)
    emotion = emotions[np.argmax(pred)]
    print(f"Predicted emotion: {emotion}")

# Example usage:
predict_emotion('test\happy\OAF_five_happy.wav')
