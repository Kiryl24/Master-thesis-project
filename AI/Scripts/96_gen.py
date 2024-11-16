import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


source_audio_folder = "WAV"
mel_output_folder = "Train_data/96Mels"

soft_folder = os.path.join(mel_output_folder, "soft")
bright_folder = os.path.join(mel_output_folder, "bright")
generic_folder = os.path.join(mel_output_folder, "generic")

os.makedirs(soft_folder, exist_ok=True)
os.makedirs(bright_folder, exist_ok=True)
os.makedirs(generic_folder, exist_ok=True)

def save_mel_spectrogram(audio_file, output_path):
    y, sr = librosa.load(audio_file, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=96)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    
    plt.imsave(output_path, mel_spectrogram_db, cmap='gray', dpi=96)
    img = Image.open(output_path).resize((96, 96))  
    img.save(output_path)

label_mapping = {'bright': bright_folder, 'soft': soft_folder, 'generic': generic_folder}


for label in ['bright', 'soft', 'generic']:
    source_folder = os.path.join(source_audio_folder, label)

    if os.path.exists(source_folder):
        for audio_file in os.listdir(source_folder):
            if audio_file.endswith(".wav"):
                audio_path = os.path.join(source_folder, audio_file)
                filename = os.path.splitext(audio_file)[0]

                mel_output_path = os.path.join(label_mapping[label], f"{filename}_mel.png")
                save_mel_spectrogram(audio_path, mel_output_path)

                print(f"Processed {audio_file}: Mel-spectrogram saved.")

print("Processing completed. Mel-spectrograms saved as 96x96 PNG images.")
