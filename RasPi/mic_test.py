import os
import time
import numpy as np
import sounddevice as sd
import librosa
import matplotlib.pyplot as plt


def record_audio(duration=4, sample_rate=22050):
    print("Recording for 4 seconds...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return audio_data.flatten()


def create_mel_spectrogram(audio, sample_rate=22050):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def display_spectrogram(mel_spec):
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(mel_spec, cmap='gray', aspect='auto')
    plt.show()


while True:
    print("Starting recording...")

    # Timer for countdown
    for i in range(3, 0, -1):
        print(i)
        time.sleep(1)

    # Record audio
    audio_data = record_audio()

    # Create mel spectrogram
    mel_spec = create_mel_spectrogram(audio_data)

    # Display mel spectrogram
    display_spectrogram(mel_spec)

    print("Ready for the next recording...\n")
