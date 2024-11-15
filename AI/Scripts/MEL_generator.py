import os
import librosa
import numpy as np
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Gain, RoomSimulator
import matplotlib.pyplot as plt
import soundfile as sf
from PIL import Image

source_audio_folder = "WAV"
mel_output_folder = "Train_data/MEL"


dark_folder = os.path.join(mel_output_folder, "dark")
bright_folder = os.path.join(mel_output_folder, "bright")
generic_folder = os.path.join(mel_output_folder, "generic")

os.makedirs(dark_folder, exist_ok=True)
os.makedirs(bright_folder, exist_ok=True)
os.makedirs(generic_folder, exist_ok=True)


augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.25, leave_length_unchanged=True),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.25),
    Gain(min_gain_in_db=-10, max_gain_in_db=10, p=0.25),
    RoomSimulator(p=0.25, leave_length_unchanged=True),
])

def save_mel_spectrogram(audio_file, output_path):
    y, sr = librosa.load(audio_file, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    
    plt.imsave(output_path, mel_spectrogram_db, cmap='viridis', dpi=128)
    img = Image.open(output_path).resize((128, 128))
    img.save(output_path)


label_mapping = {'bright': bright_folder, 'dark': dark_folder, 'generic': generic_folder}


for label in ['bright', 'dark', 'generic']:
    source_folder = os.path.join(source_audio_folder, label)
    count = 0  

    if os.path.exists(source_folder):
        for audio_file in os.listdir(source_folder):
            if audio_file.endswith(".wav"):
                
                if count >= 900:
                    break

                audio_path = os.path.join(source_folder, audio_file)
                filename = os.path.splitext(audio_file)[0]

                
                y, sr = librosa.load(audio_path, sr=None)
                augmented_y = augment(samples=y, sample_rate=sr)

                
                temp_wav = "temp.wav"
                sf.write(temp_wav, augmented_y, sr)

                
                mel_output_path = os.path.join(label_mapping[label], f"{filename}_mel.png")
                save_mel_spectrogram(temp_wav, mel_output_path)
                os.remove(temp_wav)

                count += 1  
                print(f"Processed {audio_file}: Mel-spectrogram saved ({count}/{900}).")

print("Processing completed. Mel-spectrograms saved as 128x128 PNG images.")
