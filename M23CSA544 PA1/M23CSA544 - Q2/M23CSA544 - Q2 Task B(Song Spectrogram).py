import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


def generate_spectrogram(Title,audio_path, n_fft=2048, hop_length=512):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)

    # Compute STFT with Hann window
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')

    # Convert to decibels for visualization
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Plot spectrogram
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(D_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram using STFT (Hann Window) for " + Title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.show()


# Call for the Song
base_path = r"C:\Users\Ritesh Lamba\PycharmProjects\Speech Understanding\Songs"

# CLASSIC
audio_file = base_path + r"\Classical - Lag Ja Gale Se Phir - Woh Kaun Thi  (1964).mp3"
generate_spectrogram(r"Classic Song - Lag Ja Gale", audio_file)

# JAZZ
audio_file = base_path + r"\Jazz - Kaisi Paheli Zindagani - Parineeta.mp3"
generate_spectrogram(r"Jazz Song - Kaisi Paheli", audio_file)

# ROCK
audio_file = base_path + r"\Rock - Rock On!! - Rock On!!.mp3"
generate_spectrogram(r"Rock Song - Rock On!!", audio_file)

# ELECTRONIC
audio_file = base_path + r"\Electronic - Malang (Title Track) - Malang - Unleash The Madness.mp3"
generate_spectrogram(r"Electronic Song - Malang", audio_file)