# SNN architecture for audio word recognition
# Mirrors SNNGestureNet: LIF neurons, surrogate gradients, non-spiking readout
# Input: rate-encoded mel spectrogram (time_steps, batch, 1, N_MELS=40, T=101)

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

SLOPE = 25  # surrogate sigmoid steepness -- same as gesture project


class SNNAudioNet(nn.Module):
    """
    Spiking CNN for word recognition from rate-encoded mel spectrograms.

    Input shape: (time_steps, batch, 1, N_MELS, T)
      - 1 channel (single mel spectrogram)
      - N_MELS=40 frequency bins
      - T=101 time frames (1s audio at 16kHz, 10ms hop)

    Output: logits shape (batch, num_classes)

    Architecture mirrors SNNGestureNet: three spiking conv blocks with pooling,
    global average pool, one spiking FC, non-spiking readout accumulated over time.
    """

    def __init__(self, num_classes=36, beta=0.9):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=SLOPE)

        # Conv block 1: 1 -> 16 channels, (40, 101) -> (20, 50)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.lif1  = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool1 = nn.AvgPool2d(2)

        # Conv block 2: 16 -> 32 channels, (20, 50) -> (10, 25)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.lif2  = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool2 = nn.AvgPool2d(2)

        # Conv block 3: 32 -> 64 channels, (10, 25) -> (5, 12)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.lif3  = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool3 = nn.AvgPool2d(2)

        # Global average pool: (64, 5, 12) -> 64
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Spiking FC: 64 -> 128
        self.fc1  = nn.Linear(64, 128)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # Non-spiking linear readout -- accumulated over time steps
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        x: (time_steps, batch, 1, N_MELS, T)
        Returns logits: (batch, num_classes)
        """
        time_steps = x.shape[0]

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        mem_out_accum = None

        for t in range(time_steps):
            frame = x[t]  # (batch, 1, N_MELS, T)

            out = self.pool1(self.conv1(frame))
            spk1, mem1 = self.lif1(out, mem1)

            out = self.pool2(self.conv2(spk1))
            spk2, mem2 = self.lif2(out, mem2)

            out = self.pool3(self.conv3(spk2))
            spk3, mem3 = self.lif3(out, mem3)

            out = self.gap(spk3).flatten(1)  # (batch, 64)
            out = self.fc1(out)
            spk4, mem4 = self.lif4(out, mem4)

            logits = self.fc_out(spk4)
            if mem_out_accum is None:
                mem_out_accum = logits
            else:
                mem_out_accum = mem_out_accum + logits

        return mem_out_accum  # (batch, num_classes)
