import gdown
import os
import torch
import torchaudio
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice
import re
import warnings
warnings.filterwarnings("ignore")

def split_text(text, max_words=40):
    # Break into sentences/phrases based on punctuation
    sentences = re.split(r'(?<=[।.?!])\s+', text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_len + word_count <= max_words:
            current_chunk.append(sentence)
            current_len += word_count
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_len = word_count

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


# Step 0: Make the Audio compatible
waveform, sr = torchaudio.load("M23CSA544_VoiceSample.wav")
# Resample to 24kHz if needed
resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=24000)
resampled_waveform = resampler(waveform)
# Convert to mono
if resampled_waveform.shape[0] > 1:
    resampled_waveform = resampled_waveform.mean(dim=0, keepdim=True)
# Save in correct format
torchaudio.save("voice_sample.wav", resampled_waveform, 24000, encoding="PCM_S", bits_per_sample=16)

# Step 1: Load Tortoise model
tts = TextToSpeech()

# Step 2: Load own voice
# Load waveform as tensor
waveform, sr = torchaudio.load("voice_sample.wav")
if sr != 24000:
    resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=24000)
    waveform = resample(waveform)
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)
waveform = waveform.to(tts.device)  # Move to model's device
voice_samples = [waveform]
# Call get_conditioning_latents() with waveforms instead of file paths
conditioning_latents = tts.get_conditioning_latents(voice_samples)

# Step 3: Hindi text to synthesize
# Load text from a file
with open("M23CSA544_translated_hindi_Short.txt", "r", encoding="utf-8") as f:
    text = f.read().strip()

# Step 4: Generate audio in own voice
# Step 4.1: Split into smaller chunks
segments = split_text(text, max_words=40)  # adjust as needed
# Step 4.2: Run TTS on each segment
all_audios = []
for i, segment in enumerate(segments):
    print(f"Generating segment {i+1}/{len(segments)}: {segment[:50]}...")
    audio = tts.tts_with_preset(
        segment,
        voice_samples=voice_samples,
        conditioning_latents=conditioning_latents,
        preset="fast"
    )
    all_audios.append(audio)
# Step 4.3: Concatenate all generated audio
final_audio = torch.cat(all_audios, dim=-1)

# Step 5: Save output
torchaudio.save("M23CSA544_SU_Major-Q1_hindi_tts_own_voice.wav", final_audio.squeeze(0).cpu(), 24000)
print("TTS audio generated in own voice.")
