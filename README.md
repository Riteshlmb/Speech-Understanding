# Speech-Understanding
# Overview
This project implements a Speech Emotion Recognition (SER) system to classify emotions from spoken language using the RAVDESS dataset . The system extracts acoustic features such as MFCCs , Chroma , Spectral Contrast , and Mel-Spectrogram from audio files and uses a deep learning model to predict emotions like happiness, sadness, anger, and more.

# Key Features
Feature extraction using librosa. Multi-input deep learning model combining Mel-Spectrograms and other features. Evaluation metrics: Accuracy, F1-Score, and Confusion Matrix. Visualizations for better insights into model performance. Dataset The project uses the RAVDESS dataset for Speech Emotion Recognition.

# Dataset Details
Source : RAVDESS Dataset Structure : Contains .wav files with corresponding emotion labels. Emotions include: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised. Each file is labeled with an emotion code in its filename (e.g., 03 for "Happy"). Preprocessing : Audio files are preprocessed to extract features like MFCCs, Chroma, Spectral Contrast, and Mel-Spectrogram. Features are padded/truncated to a fixed length for consistency. How to Obtain the Dataset Visit the official RAVDESS dataset page: RAVDESS Dataset on Zenodo . Download the dataset (Audio-only or Audio-Video versions). Extract the dataset and place it in the data/ directory. Dataset Organization After extraction, organize the dataset as follows:

data/ ├── Actor_01/ │ ├── 03-01-01-01-01-01-01.wav │ ├── 03-01-02-01-01-01-01.wav │ └── ... ├── Actor_02/ │ ├── 03-01-01-01-01-01-02.wav │ ├── 03-01-02-01-01-01-02.wav │ └── ... └── ... Each filename encodes metadata about the file:

Example : 03-01-02-01-01-01-01.wav 03: Modality (03 = audio-only). 01: Vocal channel (01 = speech). 02: Emotion (02 = calm). Remaining fields encode intensity, statement, repetition, and actor ID.
