'''
Load Pre-trained Speaker Verification Model
We will use WavLM Base Plus out of the four models.
'''

import torchaudio
from transformers import WavLMForXVector, Wav2Vec2Processor
import torch

from Part2 import train_loader

# Load the pre-trained model
model_name = "microsoft/wavlm-base-plus"
model = WavLMForXVector.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

'''
Load VoxCeleb1 Test Trial Pairs
The VoxCeleb1 (cleaned) trial pairs file can be found at "https://mm.kaist.ac.kr/datasets/voxceleb/".
Download and extract the trial list into vox1_test_wav/
'''
# Load the trial pairs file
trial_pairs_file = "D:/MTech(AI)-IIT/Semester-3/Speech-Understanding/M23CSA544-PA2/Speech2025Datasets/vox1/veri_test.txt"

trial_pairs = []
with open(trial_pairs_file, 'r') as f:
    for line in f.readlines():
        label, spk1_path, spk2_path = line.strip().split()
        trial_pairs.append((label, spk1_path, spk2_path))

'''
Compute Speaker Embeddings
We need to extract embeddings for each speaker and compute the cosine similarity.
'''
import librosa
import numpy as np
from torch.nn.functional import cosine_similarity


def extract_embedding(audio_path):
    waveform, sample_rate = librosa.load(audio_path, sr=16000)
    input_values = processor(waveform, return_tensors="pt", sampling_rate=16000).input_values.to(device)

    with torch.no_grad():
        embeddings = model(input_values).embeddings

    return embeddings


# Evaluate model on trial pairs
scores, labels = [], []

for label, spk1_path, spk2_path in trial_pairs:
    emb1 = extract_embedding(
        f"D:/MTech(AI)-IIT/Semester-3/Speech-Understanding/M23CSA544-PA2/Speech2025Datasets/vox1/vox1_test_wav/{spk1_path}")
    emb2 = extract_embedding(
        f"D:/MTech(AI)-IIT/Semester-3/Speech-Understanding/M23CSA544-PA2/Speech2025Datasets/vox1/vox1_test_wav/{spk2_path}")

    # Compute cosine similarity
    score = cosine_similarity(emb1, emb2).cpu().numpy().mean()
    scores.append(score)
    labels.append(int(label))  # Convert '1'/'0' to integer

'''
Compute EER, TAR@1%FAR, and Accuracy
'''
from sklearn.metrics import roc_curve, auc

# Compute EER (Equal Error Rate)
fpr, tpr, thresholds = roc_curve(labels, scores)
eer_index = np.nanargmin(np.abs(fpr - (1 - tpr)))
eer = (fpr[eer_index] + (1 - tpr[eer_index])) / 2 * 100

# TAR @ 1% FAR
far_1_idx = np.where(fpr < 0.01)[0][-1]
tar_1 = tpr[far_1_idx] * 100

# Speaker Identification Accuracy
threshold = thresholds[eer_index]
predictions = [1 if score >= threshold else 0 for score in scores]
accuracy = np.mean(np.array(predictions) == np.array(labels)) * 100

print(f"EER: {eer:.2f}%")
print(f"TAR@1%FAR: {tar_1:.2f}%")
print(f"Speaker Identification Accuracy: {accuracy:.2f}%")

'''
Preparing the VoxCeleb2 Dataset
The dataset is located in:
"D:\MTech(AI)-IIT\Semester-3\Speech-Understanding\M23CSA544-PA2\Speech2025Datasets\vox2"
Audio files: vox2_test_aac/
Metadata files: vox2_test_txt/
'''
'''
Step 1: Extract the First 100 Speakers for Training
'''
import os
import librosa

# Define dataset path
vox2_path = "D:/MTech(AI)-IIT/Semester-3/Speech-Understanding/M23CSA544-PA2/Speech2025Datasets/vox2/vox2_test_aac"

# Get sorted list of speakers
all_speakers = sorted(os.listdir(vox2_path))
train_speakers = all_speakers[:100]  # First 100 identities
test_speakers = all_speakers[100:118]  # Next 18 identities

print(f"Training Speakers: {len(train_speakers)}, Testing Speakers: {len(test_speakers)}")

'''
Applying LoRA for Efficient Fine-Tuning
LoRA reduces the number of trainable parameters, making fine-tuning computationally efficient.
'''
'''
Apply LoRA to WavLM Model
'''
from peft import LoraConfig, get_peft_model
from transformers import WavLMForXVector

# Load base model
model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus")

# Configure LoRA
lora_config = LoraConfig(
    r=8,  # Rank
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["encoder.layers"],  # Only apply LoRA to transformer layers
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

'''
Implementing ArcFace Loss
ArcFace helps in better class separation for speaker verification.
'''
# Define ArcFace Loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import Accuracy

class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_size, num_classes, scale=30.0, margin=0.5):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        theta = torch.acos(cosine)
        theta_margin = theta + self.margin
        logits = torch.cos(theta_margin) * self.scale
        return F.cross_entropy(logits, labels)

# Initialize ArcFace loss
num_classes = len(train_speakers)  # Number of unique speakers
arcface_loss = ArcFaceLoss(embedding_size=768, num_classes=num_classes).cuda()
accuracy = Accuracy(task="multiclass", num_classes=num_classes).cuda()

'''
Fine-Tuning the Model
'''
# Training Loop
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Define optimizer
optimizer = AdamW(model.parameters(), lr=3e-5)

# Move model to GPU
model.to("cuda")
model.train()

num_epochs = 5  # Training for 5 epochs

for epoch in range(num_epochs):
    total_loss, total_acc = 0, 0

    for audio_file, label in train_loader:
        audio_input = processor(audio_file, return_tensors="pt", sampling_rate=16000).input_values.cuda()
        label = label.cuda()

        optimizer.zero_grad()
        embeddings = model(audio_input).embeddings
        loss = arcface_loss(embeddings, label)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {total_loss:.4f}")

'''
Evaluating the Fine-Tuned Model
Once the model is fine-tuned, we compare it with the pre-trained version.
'''
def evaluate_speaker_verification(model, trial_pairs):
    scores, labels = [], []

    for label, spk1_path, spk2_path in trial_pairs:
        emb1 = extract_embedding(f"D:/MTech(AI)-IIT/.../{spk1_path}")
        emb2 = extract_embedding(f"D:/MTech(AI)-IIT/.../{spk2_path}")

        score = cosine_similarity(emb1, emb2).cpu().numpy().mean()
        scores.append(score)
        labels.append(int(label))

    fpr, tpr, thresholds = roc_curve(labels, scores)
    eer = (fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))] + (1 - tpr[np.nanargmin(np.abs(fpr - (1 - tpr)))])) / 2 * 100
    tar_1 = tpr[np.where(fpr < 0.01)[0][-1]] * 100

    print(f"Fine-Tuned Model: EER: {eer:.2f}%, TAR@1%FAR: {tar_1:.2f}%")

# Evaluate fine-tuned model
evaluate_speaker_verification(model, trial_pairs)

'''
Step 3: Creating a Multi-Speaker Dataset (VoxCeleb2)
Now, we need to generate a multi-speaker scenario dataset by mixing utterances from
two different speakers in the VoxCeleb2 dataset.
'''
# Extracting the First 50 Identities for Training & Next 50 for Testing
import os

# Define dataset path
vox2_path = "D:/MTech(AI)-IIT/Semester-3/Speech-Understanding/M23CSA544-PA2/Speech2025Datasets/vox2/vox2_test_aac"

# Get sorted list of speakers
all_speakers = sorted(os.listdir(vox2_path))
train_speakers = all_speakers[:50]  # First 50 identities
test_speakers = all_speakers[50:100]  # Next 50 identities

print(f"Training Speakers: {len(train_speakers)}, Testing Speakers: {len(test_speakers)}")

'''
Selecting & Loading Random Utterances
We will randomly select two different speakers and mix their utterances.
'''
import random
import librosa
import numpy as np


def select_random_utterances(speaker_list, num_pairs=100):
    """Selects random pairs of utterances from different speakers."""
    speaker_pairs = []

    for _ in range(num_pairs):
        spk1, spk2 = random.sample(speaker_list, 2)  # Pick two random speakers
        spk1_files = librosa.util.find_files(os.path.join(vox2_path, spk1))
        spk2_files = librosa.util.find_files(os.path.join(vox2_path, spk2))

        if spk1_files and spk2_files:
            spk1_utt = random.choice(spk1_files)
            spk2_utt = random.choice(spk2_files)
            speaker_pairs.append((spk1_utt, spk2_utt))

    return speaker_pairs


# Generate training and testing speaker pairs
train_pairs = select_random_utterances(train_speakers, num_pairs=500)
test_pairs = select_random_utterances(test_speakers, num_pairs=200)

print(f"Generated {len(train_pairs)} training pairs and {len(test_pairs)} testing pairs.")

'''
Mixing Two Utterances to Create Multi-Speaker Audio
We mix two utterances using SNR-controlled additive mixing (similar to LibriMix).
'''
def mix_audio(file1, file2, snr=0):
    """Mixes two audio files at a given SNR level."""
    y1, sr1 = librosa.load(file1, sr=16000)
    y2, sr2 = librosa.load(file2, sr=16000)

    # Make sure both signals are of the same length
    min_len = min(len(y1), len(y2))
    y1, y2 = y1[:min_len], y2[:min_len]

    # Compute power of signals
    power_y1 = np.mean(y1 ** 2)
    power_y2 = np.mean(y2 ** 2)

    # Scale second signal to achieve desired SNR
    scale_factor = np.sqrt((power_y1 / power_y2) * 10 ** (-snr / 10))
    y2_scaled = y2 * scale_factor

    # Mix the two signals
    mixed_audio = y1 + y2_scaled
    return mixed_audio, y1, y2, sr1

# Example usage
mixed, clean1, clean2, sr = mix_audio(train_pairs[0][0], train_pairs[0][1], snr=0)

'''
Saving the Mixed Audio Dataset
We save the mixed, clean, and speaker-separated audio.
'''
import soundfile as sf

output_dir = "D:/MTech(AI)-IIT/Semester-3/Speech-Understanding/MultiSpeakerDataset/"
os.makedirs(output_dir, exist_ok=True)

def save_mixed_dataset(speaker_pairs, dataset_type="train"):
    """Creates and saves a mixed dataset."""
    dataset_path = os.path.join(output_dir, dataset_type)
    os.makedirs(dataset_path, exist_ok=True)

    for i, (file1, file2) in enumerate(speaker_pairs):
        mixed_audio, clean1, clean2, sr = mix_audio(file1, file2, snr=0)

        # Save mixed and clean files
        sf.write(os.path.join(dataset_path, f"mixed_{i}.wav"), mixed_audio, sr)
        sf.write(os.path.join(dataset_path, f"clean1_{i}.wav"), clean1, sr)
        sf.write(os.path.join(dataset_path, f"clean2_{i}.wav"), clean2, sr)

        if i % 50 == 0:
            print(f"Saved {i} samples in {dataset_type} set.")

# Generate datasets
save_mixed_dataset(train_pairs, "train")
save_mixed_dataset(test_pairs, "test")

'''
Step 4: Speaker Separation & Speech Enhancement (SepFormer)
Now that we have created a multi-speaker dataset, we will use the SepFormer model
from SpeechBrain to separate mixed speech into individual speakers and enhance the audio quality.
'''
import torchaudio
import torch
import librosa
import numpy as np
from speechbrain.pretrained import SepformerSeparation as separator
import soundfile as sf
from pesq import pesq
from pystoi import stoi

# Perform Speaker Separation
def separate_speakers(mixed_audio_path, output_dir):
    """Separates speakers using SepFormer and saves the output."""
    # Load mixed audio
    mixed_signal, sr = torchaudio.load(mixed_audio_path)

    # Run separation
    separated_signals = model.separate_batch(mixed_signal.unsqueeze(0))

    # Convert to NumPy
    spk1, spk2 = separated_signals[0][0].numpy(), separated_signals[0][1].numpy()

    # Save output
    sf.write(f"{output_dir}/sep_spk1.wav", spk1, sr)
    sf.write(f"{output_dir}/sep_spk2.wav", spk2, sr)

    return spk1, spk2, sr


# Example usage
mixed_file = "D:/MTech(AI)-IIT/Semester-3/Speech-Understanding/MultiSpeakerDataset/test/mixed_0.wav"
output_folder = "D:/MTech(AI)-IIT/Semester-3/Speech-Understanding/SeparatedAudio/"
spk1, spk2, sr = separate_speakers(mixed_file, output_folder)

'''
Evaluate Speaker Separation Performance
We evaluate SIR, SAR, SDR, and PESQ to measure separation quality.
'''
from mir_eval.separation import bss_eval_sources


def evaluate_separation(reference1, reference2, estimate1, estimate2):
    """Computes SIR, SAR, SDR, and PESQ scores for speaker separation."""

    # Stack references and estimates for evaluation
    ref = np.vstack((reference1, reference2))
    est = np.vstack((estimate1, estimate2))

    # Compute SDR, SIR, SAR using mir_eval
    sdr, sir, sar, _ = bss_eval_sources(ref, est)

    # Compute PESQ for quality assessment
    pesq_score1 = pesq(sr, reference1, estimate1, "wb")
    pesq_score2 = pesq(sr, reference2, estimate2, "wb")

    return sdr, sir, sar, (pesq_score1 + pesq_score2) / 2


# Load original clean speech for comparison
clean1, sr = librosa.load("D:/MTech(AI)-IIT/Semester-3/Speech-Understanding/MultiSpeakerDataset/test/clean1_0.wav",
                          sr=16000)
clean2, sr = librosa.load("D:/MTech(AI)-IIT/Semester-3/Speech-Understanding/MultiSpeakerDataset/test/clean2_0.wav",
                          sr=16000)

# Evaluate
sdr, sir, sar, pesq_score = evaluate_separation(clean1, clean2, spk1, spk2)

print(f"SDR: {sdr}, SIR: {sir}, SAR: {sar}, PESQ: {pesq_score}")

'''
Speaker Identification on Separated Speech
Now that we have separated the speakers using SepFormer, 
the next step is to identify which speaker corresponds to which separated speech segment.
'''
'''
Load the Pre-Trained & Fine-Tuned Speaker Identification Models
We will use the HuBERT Large, Wav2Vec2 XLSR, UniSpeech SAT, or WavLM Base Plus models for speaker verification.
'''
from transformers import WavLMForXVector, WavLMProcessor
import torch

# Load pre-trained WavLM model and processor
processor = WavLMProcessor.from_pretrained("microsoft/wavlm-base-plus")
model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus")
model.eval()
'''
Extract Speaker Embeddings
We will extract speaker embeddings for each separated speech file 
and compare them with reference embeddings.
'''

def get_speaker_embedding(audio_path):
    """Extracts speaker embedding from an audio file."""
    # Load audio
    speech, sr = torchaudio.load(audio_path)

    # Preprocess
    inputs = processor(speech.squeeze(0), sampling_rate=sr, return_tensors="pt")

    # Forward pass to get embedding
    with torch.no_grad():
        embeddings = model(**inputs).embeddings

    return embeddings.squeeze().numpy()


# Example: Extract embeddings for separated speakers
sep_spk1_embedding = get_speaker_embedding(
    "D:/MTech(AI)-IIT/Semester-3/Speech-Understanding/SeparatedAudio/sep_spk1.wav")
sep_spk2_embedding = get_speaker_embedding(
    "D:/MTech(AI)-IIT/Semester-3/Speech-Understanding/SeparatedAudio/sep_spk2.wav")

'''
Compare with Reference Speakers (Cosine Similarity)
To identify the speaker, we compare the separated speech embeddings with reference speaker embeddings.
'''
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load reference speaker embeddings (precomputed from VoxCeleb1/VoxCeleb2)
reference_embeddings = np.load("D:/MTech(AI)-IIT/Semester-3/Speech-Understanding/reference_speaker_embeddings.npy")
reference_speakers = np.load("D:/MTech(AI)-IIT/Semester-3/Speech-Understanding/reference_speaker_labels.npy")

def identify_speaker(embedding, ref_embeddings, ref_labels):
    """Finds the closest speaker in the reference set using cosine similarity."""
    similarities = cosine_similarity([embedding], ref_embeddings)
    best_match_idx = np.argmax(similarities)
    return ref_labels[best_match_idx], similarities[0][best_match_idx]

# Identify speakers
identified_spk1, score1 = identify_speaker(sep_spk1_embedding, reference_embeddings, reference_speakers)
identified_spk2, score2 = identify_speaker(sep_spk2_embedding, reference_embeddings, reference_speakers)

print(f"Separated Speaker 1 identified as: {identified_spk1} (Confidence: {score1:.3f})")
print(f"Separated Speaker 2 identified as: {identified_spk2} (Confidence: {score2:.3f})")
'''
Compute Rank-1 Identification Accuracy
We compute the Rank-1 accuracy over the test dataset.
'''
correct = 0
total = 0

for i in range(len(test_speaker_files)):
    embedding = get_speaker_embedding(test_speaker_files[i])
    true_label = test_speaker_labels[i]
    predicted_label, _ = identify_speaker(embedding, reference_embeddings, reference_speakers)

    if predicted_label == true_label:
        correct += 1
    total += 1

rank1_accuracy = (correct / total) * 100
print(f"Rank-1 Identification Accuracy: {rank1_accuracy:.2f}%")
'''
Step 6: Novel Pipeline for Speech Separation & Identification
Now, we will combine SepFormer with the speaker identification model 
into a single pipeline for joint speech enhancement and speaker recognition.
'''
'''
Define the End-to-End Pipeline
We will create a function to process any input multi-speaker audio, separate the speakers, and identify them.
'''
import torchaudio
import numpy as np
from transformers import WavLMForXVector, WavLMProcessor
from speechbrain.pretrained import SepformerSeparation as separator
from sklearn.metrics.pairwise import cosine_similarity

# Load SepFormer model for separation
sepformer = separator.from_hparams(source="speechbrain/sepformer-whamr", savedir="pretrained_models/sepformer-whamr")

# Load fine-tuned speaker verification model
processor = WavLMProcessor.from_pretrained("microsoft/wavlm-base-plus")
model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus")
model.eval()

# Load reference speaker embeddings
reference_embeddings = np.load("D:/MTech(AI)-IIT/Semester-3/Speech-Understanding/reference_speaker_embeddings.npy")
reference_speakers = np.load("D:/MTech(AI)-IIT/Semester-3/Speech-Understanding/reference_speaker_labels.npy")


def process_audio(mixed_audio_path):
    """Separates speakers and identifies them."""

    # Step 1: Separate speakers using SepFormer
    est_sources = sepformer.separate_file(mixed_audio_path)
    sep_spk1, sep_spk2 = est_sources[0], est_sources[1]

    # Step 2: Extract embeddings and identify speakers
    def get_speaker_embedding(audio_tensor):
        """Extract speaker embeddings."""
        inputs = processor(audio_tensor.squeeze(0), sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            embeddings = model(**inputs).embeddings
        return embeddings.squeeze().numpy()

    spk1_embedding = get_speaker_embedding(sep_spk1)
    spk2_embedding = get_speaker_embedding(sep_spk2)

    def identify_speaker(embedding):
        """Identify speaker based on closest reference embedding."""
        similarities = cosine_similarity([embedding], reference_embeddings)
        best_match_idx = np.argmax(similarities)
        return reference_speakers[best_match_idx], similarities[0][best_match_idx]

    # Identify speakers
    identified_spk1, score1 = identify_speaker(spk1_embedding)
    identified_spk2, score2 = identify_speaker(spk2_embedding)

    return identified_spk1, identified_spk2, score1, score2, sep_spk1, sep_spk2

'''
Evaluate on Multi-Speaker Test Set
We will test the pipeline on multi-speaker mixed audio files.
'''
correct = 0
total = 0
sir_scores, sar_scores, sdr_scores, pesq_scores = [], [], [], []

for test_file, true_spk1, true_spk2 in test_data:  # test_data contains (file_path, true_speaker_1, true_speaker_2)
    pred_spk1, pred_spk2, score1, score2, sep1, sep2 = process_audio(test_file)

    # Compute Rank-1 accuracy
    if pred_spk1 in [true_spk1, true_spk2]:
        correct += 1
    if pred_spk2 in [true_spk1, true_spk2]:
        correct += 1
    total += 2

    # Compute SIR, SAR, SDR, PESQ
    sir, sar, sdr = compute_separation_metrics(test_file, sep1, sep2)  # Implement metric function
    pesq = compute_pesq(test_file, sep1, sep2)

    sir_scores.append(sir)
    sar_scores.append(sar)
    sdr_scores.append(sdr)
    pesq_scores.append(pesq)

rank1_accuracy = (correct / total) * 100

# Report results
print(f"Rank-1 Speaker Identification Accuracy: {rank1_accuracy:.2f}%")
print(f"Avg SIR: {np.mean(sir_scores):.2f}, Avg SAR: {np.mean(sar_scores):.2f}, Avg SDR: {np.mean(sdr_scores):.2f}, Avg PESQ: {np.mean(pesq_scores):.2f}")
