"""Streamlit interface for the speech emotion recognition model."""

from pathlib import Path
import tempfile

import streamlit as st
from tensorflow.keras.models import load_model

from src.features import extract_mfcc
from src.predict import emotions

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'emotion_cnn.h5'

EMOTION_META = {
    'happy': ('😊', 'Happy'),
    'angry': ('😠', 'Angry'),
    'sad': ('😔', 'Sad'),
}


@st.cache_resource(show_spinner=False)
def get_model():
    return load_model(MODEL_PATH)


def classify(audio_bytes: bytes, suffix: str):
    """Extract features from an upload and return class probabilities."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as file:
        file.write(audio_bytes)
        temp_path = Path(file.name)
    try:
        features = extract_mfcc(temp_path)[None, ..., None]
        return get_model().predict(features, verbose=0)[0]
    finally:
        temp_path.unlink(missing_ok=True)


st.set_page_config(page_title='Voice Emotion', page_icon='🎙️', layout='centered')
st.title('🎙️ Voice Emotion Recognition')
st.caption('Upload a short WAV recording and the trained model will identify the emotion in the speech.')

if not MODEL_PATH.exists():
    st.error('The trained model file `emotion_cnn.h5` was not found beside this app.')
    st.stop()

audio_file = st.file_uploader('Choose a WAV audio file', type=['wav'])

if audio_file:
    st.audio(audio_file, format='audio/wav')
    if st.button('Analyze emotion', type='primary', use_container_width=True):
        try:
            with st.spinner('Listening to the audio…'):
                probabilities = classify(audio_file.getvalue(), Path(audio_file.name).suffix or '.wav')
            best_index = int(probabilities.argmax())
            emotion = emotions[best_index]
            icon, label = EMOTION_META.get(emotion, ('🎵', emotion.title()))
            st.success(f'{icon} Predicted emotion: **{label}**')
            st.metric('Model confidence', f'{probabilities[best_index]:.1%}')
            st.subheader('All predictions')
            st.bar_chart({EMOTION_META.get(name, ('', name.title()))[1]: float(probabilities[i])
                          for i, name in enumerate(emotions)})
        except Exception as error:
            st.error(f'Could not analyze this file: {error}')
else:
    st.info('Supported format: WAV. For best results, use clear speech similar to the training recordings.')
