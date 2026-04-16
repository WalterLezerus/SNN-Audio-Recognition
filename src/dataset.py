# Dataset loading and spike encoding for Google Speech Commands
# Audio pipeline: waveform -> mel spectrogram -> rate-encoded spike trains
#
# Uses soundfile for WAV loading to avoid torchaudio's torchcodec/FFmpeg dependency.
# torchaudio.transforms (MelSpectrogram, AmplitudeToDB) are pure PyTorch -- no FFmpeg needed.

import os
import urllib.request
import tarfile
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import torch
from torch.utils.data import DataLoader

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
DATASET_URL = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
DATASET_FOLDER = 'SpeechCommands'

# All 35 core words in Speech Commands v2
CLASSES = sorted([
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five',
    'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left',
    'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila',
    'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
])
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

SAMPLE_RATE = 16000
N_MELS = 40
N_FFT = 512
HOP_LENGTH = 160  # 10ms hop -> T=101 frames for 1s audio


def _download(data_dir):
    """Download and extract Speech Commands v0.02 if not already present."""
    extract_dir = Path(data_dir) / DATASET_FOLDER
    # The tar contains a top-level speech_commands_v0.02/ folder
    dataset_dir = extract_dir / 'speech_commands_v0.02'

    if dataset_dir.exists():
        return dataset_dir

    tar_path = Path(data_dir) / 'speech_commands_v0.02.tar.gz'
    os.makedirs(data_dir, exist_ok=True)

    if not tar_path.exists():
        print("Downloading Speech Commands v0.02 (~2.4GB)...")
        urllib.request.urlretrieve(DATASET_URL, tar_path)
        print("Download complete.")

    print("Extracting...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        tf.extractall(extract_dir)
    print("Extraction complete.")

    return dataset_dir


def _get_samples(dataset_dir, subset):
    """
    Returns list of (wav_path, label) for the given subset.
    Splits are defined by validation_list.txt and testing_list.txt in the dataset root.
    Training = everything not in those two lists.
    """
    dataset_dir = Path(dataset_dir)
    val_set  = set((dataset_dir / 'validation_list.txt').read_text().splitlines())
    test_set = set((dataset_dir / 'testing_list.txt').read_text().splitlines())

    samples = []
    for label in CLASSES:
        label_dir = dataset_dir / label
        if not label_dir.exists():
            continue
        for wav_file in sorted(label_dir.glob('*.wav')):
            rel = f"{label}/{wav_file.name}"
            if subset == 'validation':
                if rel in val_set:
                    samples.append((str(wav_file), label))
            elif subset == 'testing':
                if rel in test_set:
                    samples.append((str(wav_file), label))
            else:  # training
                if rel not in val_set and rel not in test_set:
                    samples.append((str(wav_file), label))

    return samples


def rate_encode(mel_spec, n_steps):
    """
    Convert a normalized mel spectrogram to a spike train via rate coding.

    mel_spec: (1, N_MELS, T) float tensor in [0, 1]
    n_steps:  number of independent Bernoulli samples to draw

    Returns: (n_steps, 1, N_MELS, T) binary tensor
    """
    mel_expanded = mel_spec.unsqueeze(0).expand(n_steps, -1, -1, -1)
    return torch.bernoulli(mel_expanded)


class SpeechCommandsDataset(torch.utils.data.Dataset):
    def __init__(self, subset, n_time_steps=50, data_dir=DATA_DIR):
        """
        subset: 'training', 'validation', or 'testing'
        n_time_steps: number of rate-coded spike frames to generate per sample
        """
        self.n_time_steps = n_time_steps
        dataset_dir = _download(data_dir)
        self.samples = _get_samples(dataset_dir, subset)
        print(f"[{subset}] {len(self.samples)} samples loaded.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, label = self.samples[idx]

        # soundfile reads (samples,) as numpy; Speech Commands is mono
        audio, sample_rate = sf.read(wav_path, dtype='float32')

        # Pad or trim to exactly 1 second
        if len(audio) < SAMPLE_RATE:
            audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)))
        else:
            audio = audio[:SAMPLE_RATE]

        # Mel spectrogram via librosa: (N_MELS, T=101)
        mel = librosa.feature.melspectrogram(
            y=audio, sr=SAMPLE_RATE, n_fft=N_FFT,
            hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        mel = librosa.power_to_db(mel)  # log scale

        # Normalize to [0, 1] for Bernoulli sampling
        mel_min, mel_max = mel.min(), mel.max()
        if mel_max > mel_min:
            mel = (mel - mel_min) / (mel_max - mel_min)
        else:
            mel = np.zeros_like(mel)

        mel = torch.from_numpy(mel).unsqueeze(0)  # (1, N_MELS, T)

        # Rate encode: (n_time_steps, 1, N_MELS, T)
        spikes = rate_encode(mel, self.n_time_steps)

        return spikes, CLASS_TO_IDX[label]


def collate_fn(batch):
    """
    Stack batch and transpose to (time_steps, batch, C, H, W)
    to match the SNN forward loop convention used in the gesture project.
    """
    spikes, labels = zip(*batch)
    spikes = torch.stack(spikes)            # (batch, n_time_steps, 1, N_MELS, T)
    spikes = spikes.permute(1, 0, 2, 3, 4) # (n_time_steps, batch, 1, N_MELS, T)
    labels = torch.tensor(labels, dtype=torch.long)
    return spikes, labels


def get_dataloaders(batch_size=32, n_time_steps=50, data_dir=DATA_DIR):
    train_dataset = SpeechCommandsDataset('training',   n_time_steps, data_dir)
    val_dataset   = SpeechCommandsDataset('validation', n_time_steps, data_dir)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_fn
    )

    return train_loader, val_loader
