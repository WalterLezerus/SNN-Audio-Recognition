# Dataset loading and spike encoding for Google Speech Commands
# Audio pipeline: waveform -> mel spectrogram -> rate-encoded spike trains

import os
import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

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


def rate_encode(mel_spec, n_steps):
    """
    Convert a normalized mel spectrogram to a spike train via rate coding.

    mel_spec: (1, N_MELS, T) float tensor in [0, 1]
              each value is the spike probability for that bin at that frame
    n_steps:  number of independent spike samples to draw

    Returns: (n_steps, 1, N_MELS, T) binary tensor
    """
    # Expand along new time dimension and sample Bernoulli independently
    mel_expanded = mel_spec.unsqueeze(0).expand(n_steps, -1, -1, -1)
    return torch.bernoulli(mel_expanded)


class SpeechCommandsDataset(torch.utils.data.Dataset):
    def __init__(self, subset, n_time_steps=50, data_dir=DATA_DIR):
        """
        subset: 'training', 'validation', or 'testing'
        n_time_steps: number of rate-coded spike frames to generate per sample
        """
        self.n_time_steps = n_time_steps
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        raw = SPEECHCOMMANDS(root=data_dir, subset=subset, download=True)
        # Filter to only the 35 core word classes
        self.samples = [s for s in raw if s[2] in CLASS_TO_IDX]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        waveform, sample_rate, label, *_ = self.samples[idx]

        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)

        # Pad or trim to exactly 1 second
        n = waveform.shape[1]
        if n < SAMPLE_RATE:
            waveform = torch.nn.functional.pad(waveform, (0, SAMPLE_RATE - n))
        else:
            waveform = waveform[:, :SAMPLE_RATE]

        # Mel spectrogram: (1, N_MELS, T=101)
        mel = self.mel_transform(waveform)       # power spectrogram
        mel = self.amplitude_to_db(mel)          # log scale, more spike-friendly

        # Normalize to [0, 1] for Bernoulli sampling
        mel_min, mel_max = mel.min(), mel.max()
        if mel_max > mel_min:
            mel = (mel - mel_min) / (mel_max - mel_min)
        else:
            mel = torch.zeros_like(mel)

        # Rate encode: (n_time_steps, 1, N_MELS, T)
        spikes = rate_encode(mel, self.n_time_steps)

        return spikes, CLASS_TO_IDX[label]


def collate_fn(batch):
    """
    Stack batch and transpose to (time_steps, batch, C, H, W)
    to match the SNN forward loop convention used in the gesture project.
    """
    spikes, labels = zip(*batch)
    spikes = torch.stack(spikes)           # (batch, n_time_steps, 1, N_MELS, T)
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
