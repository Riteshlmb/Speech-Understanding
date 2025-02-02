import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Dataset paths
csv_path = r"C:\Users\Ritesh Lamba\PycharmProjects\Speech Understanding\UrbanSound8K\UrbanSound8K.csv"
base_path = r"C:\Users\Ritesh Lamba\PycharmProjects\Speech Understanding\UrbanSound8K"

# Load metadata
data = pd.read_csv(csv_path)

# STFT parameters
n_fft = 1024
hop_length = 512
window_types = {
    "Hann": np.hanning(n_fft),
    "Hamming": np.hamming(n_fft),
    "Rectangular": np.ones(n_fft),
}

# Function to plot and compare spectrograms
def plot_spectrograms(file_path, sr, y):
    plt.figure(figsize=(12, 6))
    for i, (win_name, window) in enumerate(window_types.items(), 1):
        _, _, Zxx = stft(y, fs=sr, window=window, nperseg=n_fft, noverlap=hop_length)
        spectrogram = np.abs(Zxx)

        plt.subplot(1, 3, i)
        plt.title(f"{win_name} Window")
        plt.imshow(10 * np.log10(spectrogram), aspect='auto', origin='lower', cmap='inferno')
        plt.xlabel("Time")
        plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

# Function to extract MFCC features from a spectrogram
def extract_features(file_path, window_name):
    y, sr = librosa.load(file_path, sr=None)

    # Select window function
    window = window_types[window_name]

    _, _, Zxx = stft(y, fs=sr, window=window, nperseg=n_fft, noverlap=hop_length)
    spectrogram = np.abs(Zxx)

    # Convert to MFCC features
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(spectrogram), sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)  # Take the mean across time


# Initialize storage for features and labels
features = {"Hann": [], "Hamming": [], "Rectangular": []}
labels = []

# Process dataset and extract features
for index, row in data.iterrows():
    file_path = os.path.join(base_path, f"fold{row['fold']}", row['slice_file_name'])
    try:
        for window_name in window_types.keys():
            mfcc_features = extract_features(file_path, window_name)
            features[window_name].append(mfcc_features)
        labels.append(row["classID"])
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Train and evaluate classifiers for each window type
for window_name, feature_list in features.items():
    print(f"\n🔹 Training classifier with {window_name} window...")

    # Convert to numpy arrays
    X = np.array(feature_list)
    y = np.array(labels)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train SVM classifier
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)

    # Evaluate performance
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Classification Accuracy ({window_name} Window): {accuracy:.2f}")

# Load an example audio file for spectrogram visualization
example_file = os.path.join(base_path, "fold10", data.iloc[10]['slice_file_name'])

# Debugging: Print the constructed file path
print(f"Attempting to load file: {example_file}")

# Verify if the file exists
if os.path.exists(example_file):
    print(f"File exists: {example_file}")
    y, sr = librosa.load(example_file, sr=None)

    # Visualize spectrograms if the file is loaded correctly
    plot_spectrograms(example_file, sr, y)
else:
    print(f"File not found: {example_file}")

# Generate and compare spectrograms
print("\n🔍 Visualizing spectrograms for an example audio file...")
plot_spectrograms(example_file, sr, y)