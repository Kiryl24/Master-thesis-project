import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PilImage
import soundfile as sf


def create_mel_spectrogram(audio, sample_rate=44100, title="Mel Spectrogram"):
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=224,
        fmin=100,
        fmax=10000,
        n_fft=2024,
        hop_length=512
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    mel_spec_path = os.path.join("spectrograms", "mel_spec_720x720.png")
    plt.figure(figsize=(7.2, 7.2), dpi=100)  # 720x720 pixels

    # Dodaj tytuł
    plt.title(title)

    # Wyświetl spektrogram z osiami
    librosa.display.specshow(mel_spec_db, sr=sample_rate, hop_length=512, x_axis='time', y_axis='mel', fmin=100,
                             fmax=10000, cmap='viridis')

    # Dodaj kolorową legendę
    plt.colorbar(format='%+2.0f dB')

    # Etykiety osi
    plt.xlabel('Czas (s)')
    plt.ylabel('Częstotliwość (Hz)')

    # Dodaj linię przy 440 Hz z podpisem "A4 Frequency"
    plt.axhline(y=440, color='r', linestyle='-')
    plt.text(plt.xlim()[1], plt.ylim()[1],  'A4 Frequency 440Hz', color='r', ha='right', va='top', fontsize=20,

             bbox=dict(facecolor='white', alpha=0.0, edgecolor='none'))

    # Zapisz wykres
    plt.savefig(mel_spec_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return mel_spec_path


def process_audio(file_path):
    try:
        # Wczytaj dane audio z pliku WAV
        audio_data, sample_rate = librosa.load(file_path, sr=44100)

        # Generuj spektrogram
        mel_spec_path = create_mel_spectrogram(audio_data, sample_rate, title="Mel Spectrogram")

        # Wczytaj obraz spektrogramu
        mel_image = PilImage.open(mel_spec_path)
        print(f"Image size: {mel_image.size}")

    except Exception as e:
        print(f"Error processing audio:\n{e}")


if __name__ == "__main__":
    # Upewnij się, że folder "spectrograms" istnieje
    os.makedirs("spectrograms", exist_ok=True)

    # Ścieżka do pliku WAV
    file_path = "WAV/generic/sample_1.wav"

    # Przetwarzaj audio
    process_audio(file_path)