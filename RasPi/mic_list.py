import sounddevice as sd
import soundfile as sf
import os
# Wyświetlenie dostępnych urządzeń
devices = sd.query_devices()
print(devices)


# Upewnij się, że folder 'temp' istnieje
if not os.path.exists('temp'):
    os.makedirs('temp')

# Parametry nagrywania
duration = 5  # Czas nagrywania w sekundach
sample_rate = 44100  # Częstotliwość próbkowania (Hz)
channels = 1  # Liczba kanałów (1 dla mono, 2 dla stereo)

# Funkcja nagrywająca dźwięk
def record_audio():
    print("Rozpoczynam nagrywanie...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='float32')
    sd.wait()  # Czekaj na zakończenie nagrywania
    print("Nagrywanie zakończone.")
    return audio_data

# Zapisz dźwięk do pliku WAV w folderze 'temp'
def save_audio_to_file(audio_data):
    file_path = 'temp/test.wav'
    sf.write(file_path, audio_data, sample_rate)
    print(f"Zapisano dźwięk w {file_path}")

# Nagrywamy i zapisujemy audio
audio_data = record_audio()
save_audio_to_file(audio_data)
