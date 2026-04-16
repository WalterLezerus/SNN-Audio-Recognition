# Real-time continuous word recognition using the trained SNNAudioNet
#
# Maintains a rolling 1-second audio buffer.
# Every HOP_MS milliseconds, runs inference on the current window.
# Only fires when confidence exceeds CONFIDENCE_THRESHOLD.
# Debounces to avoid repeatedly printing the same word.

from pathlib import Path
from collections import deque
import threading
import numpy as np
import sounddevice as sd
import librosa
import torch

from model import SNNAudioNet
from dataset import CLASSES, NUM_CLASSES, SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH, rate_encode

CHECKPOINT          = str(Path(__file__).parent.parent / "models" / "best.pth")
N_TIME_STEPS        = 50
CONFIDENCE_THRESHOLD = 0.90   # only report predictions above this confidence
ENERGY_THRESHOLD    = 0.005   # RMS below this is treated as silence -- tune to your environment
HOP_MS              = 200     # run inference every 200ms
DEBOUNCE_MS         = 800     # don't repeat the same word within this window

HOP_SAMPLES = int(SAMPLE_RATE * HOP_MS / 1000)


def preprocess(audio):
    """numpy float32 (SAMPLE_RATE,) -> spike tensor (N_TIME_STEPS, 1, 1, N_MELS, T)"""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel = librosa.power_to_db(mel)
    mel_min, mel_max = mel.min(), mel.max()
    if mel_max > mel_min:
        mel = (mel - mel_min) / (mel_max - mel_min)
    else:
        mel = np.zeros_like(mel)
    mel_t  = torch.from_numpy(mel).float().unsqueeze(0)  # (1, N_MELS, T)
    spikes = rate_encode(mel_t, N_TIME_STEPS)             # (N_TIME_STEPS, 1, N_MELS, T)
    return spikes.unsqueeze(1)                            # (N_TIME_STEPS, 1, 1, N_MELS, T)


def main():
    model = SNNAudioNet(num_classes=NUM_CLASSES)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()

    # Rolling buffer: always holds the last 1 second of audio
    buffer = deque(np.zeros(SAMPLE_RATE, dtype=np.float32), maxlen=SAMPLE_RATE)
    buffer_lock = threading.Lock()

    def audio_callback(indata, frames, time, status):
        with buffer_lock:
            buffer.extend(indata[:, 0])

    last_word   = None
    last_fired  = -DEBOUNCE_MS  # ms since epoch, starts ready to fire

    import time as time_mod

    print(f"Listening... (threshold: {CONFIDENCE_THRESHOLD*100:.0f}%  |  Ctrl+C to stop)\n")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype='float32', blocksize=HOP_SAMPLES,
                        callback=audio_callback):
        while True:
            time_mod.sleep(HOP_MS / 1000)

            with buffer_lock:
                window = np.array(buffer, dtype=np.float32)

            # Gate on energy -- skip inference if the window is likely silence
            rms = np.sqrt(np.mean(window ** 2))

            if rms < ENERGY_THRESHOLD:
                continue

            with torch.no_grad():
                spikes = preprocess(window)
                logits = model(spikes)
                probs  = torch.softmax(logits, dim=1)[0]
                conf, idx = probs.max(dim=0)
                word = CLASSES[idx.item()]
                conf = conf.item()

            if conf < CONFIDENCE_THRESHOLD or word == 'silence':
                continue

            now_ms = time_mod.time() * 1000
            if word == last_word and (now_ms - last_fired) < DEBOUNCE_MS:
                continue

            print(f"{word:<12} {conf*100:5.1f}%")
            last_word  = word
            last_fired = now_ms


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
