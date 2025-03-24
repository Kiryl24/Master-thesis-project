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

    mel_spec_path = os.path.join("spectrograms", "mel_spec_generic.png")
    plt.figure(figsize=(7.2, 7.2), dpi=100)  # 720x720 pixels

    # Dodaj tytuł
    plt.title(title)

    # Wyświetl spektrogram z osiami
    librosa.display.specshow(mel_spec_db, sr=sample_rate, hop_length=512, x_axis='time', y_axis='mel', fmin=100,
                             fmax=10000, cmap='viridis')

    # Dodaj kolorową legendę
    plt.colorbar(format='%+2.0f dB')
    ax = plt.gca()
    # Etykiety osi
    plt.xlabel('Czas (s)')
    plt.ylabel('Częstotliwość (Hz)')
    base_frequency = 440  # Ton podstawowy (1 rząd) - linia czerwona
    last_frequency = 7040  # Ostatnia harmoniczna (16 rząd)
    max_n = last_frequency // base_frequency  # 7040 / 440 = 16
    # Dodaj linię przy 440 Hz z podpisem "A4 Frequency"
    for n in range(1, int(max_n) + 1):
        freq = n * base_frequency
        plt.axhline(y=freq, color='white', linestyle='-', alpha=0.4)
        # Dodanie etykiety dla każdej harmonicznej
        ax.text(0.85, freq, f'n = {n}', color='black', ha='left', va='center', fontweight='bold',
                fontsize=10, transform=ax.get_yaxis_transform())

    # Zapisz wykres
    plt.savefig(mel_spec_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return mel_spec_path


def process_audio(file_path):
    try:
        # Wczytaj dane audio z pliku WAV
        audio_data, sample_rate = librosa.load(file_path, sr=44100)
        sound = "generic"
        # Generuj spektrogram
        mel_spec_path = create_mel_spectrogram(audio_data, sample_rate, title=f"Mel Spectrogram for {sound} sound with harmonic series")

        # Wczytaj obraz spektrogramu
        mel_image = PilImage.open(mel_spec_path)
        print(f"Image size: {mel_image.size}")

    except Exception as e:
        print(f"Error processing audio:\n{e}")


if __name__ == "__main__":
    # Upewnij się, że folder "spectrograms" istnieje
    os.makedirs("spectrograms", exist_ok=True)

    # Ścieżka do pliku WAV
    file_path = "WAV/generic/sample_4.wav"

    # Przetwarzaj audio
    process_audio(file_path)