# Emotion Recognition from Speech

## Overview

This project implements an emotion recognition system that classifies emotions (angry, happy, sad) from speech audio files using deep learning. The system is trained on labeled speech data and can predict the emotion expressed in new audio samples.

## Project Structure

```
Emotion Recognition from Speech/
├── data/           # Training data organized by emotion
│   ├── angry/
│   ├── happy/
│   └── sad/
├── test/           # Test data organized by emotion
│   ├── angry/
│   ├── happy/
│   └── sad/
├── src/            # Source code
│   ├── features.py     # Feature extraction from audio
│   ├── model.py        # Model architecture
│   ├── train.py        # Model training script
│   └── predict.py      # Prediction script
├── emotion_cnn.h5  # Trained model weights
├── requirements.txt    # Python dependencies
└── README.md       # Project documentation
```

## Dataset
- **data/**: Contains training audio files in WAV format, organized by emotion (angry, happy, sad).
- **test/**: Contains test audio files for evaluation, organized similarly.

## Setup

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd "Emotion Recognition from Speech"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Download or prepare additional data**
   - Place your WAV files in the appropriate `data/` or `test/` subfolders.

## Usage

### Training the Model
Run the following command to train the model:
```bash
python src/train.py
```
This will train the model and save the weights to `emotion_cnn.h5`.

### Predicting Emotions
To predict the emotion of a new audio file:
```bash
python src/predict.py <path_to_audio_file>
```
The script will output the predicted emotion.

## Feature Extraction
Feature extraction from audio is handled in `src/features.py`. You can modify this file to experiment with different audio features (e.g., MFCCs, chroma, mel spectrogram).

## Model Architecture
The model is defined in `src/model.py`. It is a convolutional neural network (CNN) designed for audio classification tasks.

## Requirements
See `requirements.txt` for the list of required Python packages (e.g., numpy, librosa, tensorflow, keras).

 