import numpy as np
from sklearn.model_selection import train_test_split
from src.features import load_data
from src.model import build_cnn

# Define your emotions (folder names in data/)
emotions = ['happy', 'angry', 'sad']  # Edit as per your dataset

# Load data
X, y = load_data('data', emotions)
X = X[..., np.newaxis]  # Add channel dimension

# Split
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

# Build model
input_shape = X_train.shape[1:]
num_classes = len(emotions)
model = build_cnn(input_shape, num_classes)

# Train
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save('emotion_cnn.h5') 