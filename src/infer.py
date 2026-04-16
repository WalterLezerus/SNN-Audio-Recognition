# Record a 1-second audio clip and run inference with the trained SNNAudioNet

from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import torch

from model import SNNAudioNet
from dataset import CLASSES, NUM_CLASSES, SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH, rate_encode

CHECKPOINT = str(Path(__file__).parent.parent / "models" / "best.pth")
N_TIME_STEPS = 50
RECORD_SECONDS = 1


def preprocess(audio):
    """numpy float32 array (N,) -> spike tensor (N_TIME_STEPS, 1, N_MELS, T)"""
    if len(audio) < SAMPLE_RATE:
        audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)))
    else:
        audio = audio[:SAMPLE_RATE]

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

    mel_t = torch.from_numpy(mel).float().unsqueeze(0)      # (1, N_MELS, T)
    spikes = rate_encode(mel_t, N_TIME_STEPS)               # (N_TIME_STEPS, 1, N_MELS, T)
    return spikes.unsqueeze(1)                               # (N_TIME_STEPS, 1, 1, N_MELS, T)


def main():
    # Load model on CPU for inference -- no need for DirectML here
    model = SNNAudioNet(num_classes=NUM_CLASSES)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()

    input(f"Press ENTER then immediately say your word (recording lasts {RECORD_SECONDS} second)...")


    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    audio = audio.flatten()

    # Optionally save the recording
    sf.write("last_recording.wav", audio, SAMPLE_RATE)

    with torch.no_grad():
        spikes = preprocess(audio)
        logits = model(spikes)                   # (1, NUM_CLASSES)
        probs  = torch.softmax(logits, dim=1)[0]
        top5   = probs.topk(5)

    print("\nTop 5 predictions:")
    for prob, idx in zip(top5.values, top5.indices):
        print(f"  {CLASSES[idx]:<12} {prob.item()*100:5.1f}%")


if __name__ == "__main__":
    main()
