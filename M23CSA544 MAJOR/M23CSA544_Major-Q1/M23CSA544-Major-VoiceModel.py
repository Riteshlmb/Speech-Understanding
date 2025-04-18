from TTS.api import TTS
import os

# Path to my voice sample
speaker_wav = "M23CSA544_VoiceSample.wav"

# English text I want to synthesize
# Load text from a file
with open("../transcript_cleaned.txt", "r", encoding="utf-8") as f:
    text = f.read().strip()

# Output file name
output_path = "my_cloned_output.wav"

# Load the YourTTS model from Coqui
print("Loading YourTTS model... (This might take a few seconds)")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=True, gpu=True)

# Synthesize speech with the voice sample
tts.tts_to_file(
    text=text,
    speaker_wav=speaker_wav,
    language="en",
    file_path=output_path
)

print(f"\n✅ Audio generated and saved to: {output_path}\n")
print("🔊 Play the file using your system's audio player to verify output.")
