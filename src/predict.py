import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path

try:
    from .features import extract_mfcc
except ImportError:  # Allows: python src/predict.py <audio_file>
    from features import extract_mfcc

emotions = ['happy', 'angry', 'sad']  # Same order as training

def predict_emotion(file_path, model_path='emotion_cnn.h5'):
    """Predict an emotion and return its label plus per-class confidence."""
    mfcc = extract_mfcc(file_path)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]  # Add batch and channel dims
    model = load_model(model_path)
    probabilities = model.predict(mfcc, verbose=0)[0]
    emotion = emotions[int(np.argmax(probabilities))]
    return emotion, dict(zip(emotions, map(float, probabilities)))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Predict emotion from a WAV audio file.')
    parser.add_argument('audio_file', type=Path)
    parser.add_argument('--model', default='emotion_cnn.h5')
    args = parser.parse_args()
    emotion, confidence = predict_emotion(args.audio_file, args.model)
    print(f"Predicted emotion: {emotion}")
    print(f"Confidence: {confidence[emotion]:.1%}")
