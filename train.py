# train.py
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import tensorflow as tf  # Required import

# Configuration
DATA_PATH = "data"
MAX_PAD_LENGTH = 120
EPOCHS = 100
BATCH_SIZE = 32
CONFIDENCE_THRESHOLD = 0.6

# Auto-detect classes
CLASSES = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])

def extract_features(file_path):
    signal, sr = librosa.load(file_path, sr=22050)
    
    # Handle very short signals
    if len(signal) < 2048:
        signal = np.pad(signal, (0, 2048 - len(signal)), mode='constant')
    
    # Dynamic n_fft selection
    n_fft = 2048
    if len(signal) < n_fft:
        n_fft = len(signal) // 2
        n_fft = n_fft if n_fft % 2 == 0 else n_fft - 1

    mfccs = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=40,
        n_fft=n_fft,
        hop_length=512
    )
    
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    
    features = np.concatenate([mfccs, delta_mfccs, delta2_mfccs])
    
    if features.shape[1] < MAX_PAD_LENGTH:
        pad_width = MAX_PAD_LENGTH - features.shape[1]
        features = np.pad(features, ((0,0), (0, pad_width)), mode='constant')
    else:
        features = features[:, :MAX_PAD_LENGTH]
    
    return features.T

# Verify data distribution
print("Data distribution:")
class_counts = {}
for cls in CLASSES:
    count = len(os.listdir(os.path.join(DATA_PATH, cls)))
    class_counts[cls] = count
    print(f"{cls}: {count} files")

# Load and preprocess data
features = []
labels = []

for cls in CLASSES:
    class_path = os.path.join(DATA_PATH, cls)
    for file in os.listdir(class_path):
        file_path = os.path.join(class_path, file)
        features.append(extract_features(file_path))
        labels.append(cls)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)
y = to_categorical(y, num_classes=len(CLASSES))

# Split data
X = np.array(features)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))

# Data augmentation
def augment_data(X, y):
    augmented_X = []
    augmented_y = []
    for i in range(len(X)):
        original = X[i]
        augmented_X.append(original)
        augmented_y.append(y[i])
        
        # Time stretch
        stretched = librosa.effects.time_stretch(original.T, rate=0.8).T
        augmented_X.append(stretched[:MAX_PAD_LENGTH, :])
        augmented_y.append(y[i])
        
        # Pitch shift
        shifted = librosa.effects.pitch_shift(original.T, sr=22050, n_steps=2).T
        augmented_X.append(shifted[:MAX_PAD_LENGTH, :])
        augmented_y.append(y[i])
        
        # Noise injection
        noise = np.random.randn(*original.shape) * 0.005
        noisy = original + noise
        augmented_X.append(noisy)
        augmented_y.append(y[i])
    
    return np.array(augmented_X), np.array(augmented_y)

X_train_aug, y_train_aug = augment_data(X_train, y_train)

# Build enhanced model
def build_model(input_shape):
    model = Sequential([
        Conv1D(128, 3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.2),
        
        Conv1D(256, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.2),
        
        LSTM(256, return_sequences=True),
        Dropout(0.3),
        
        LSTM(128),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(len(CLASSES), activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model

model = build_model((MAX_PAD_LENGTH, 120))

# Train with early stopping
history = model.fit(
    X_train_aug, y_train_aug,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            start_from_epoch=5
        )
    ]
)

# Evaluate model
y_pred = np.argmax(model.predict(X_test), axis=-1)
y_true = np.argmax(y_test, axis=-1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASSES))

# Save artifacts
model.save("model/trained_model.h5")
import joblib
joblib.dump(le, "model/label_encoder.pkl")