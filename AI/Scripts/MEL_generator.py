import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


source_audio_folder = "WAV"
mel_output_folder = "Train_data/224Mels"

soft_folder = os.path.join(mel_output_folder, "soft")
bright_folder = os.path.join(mel_output_folder, "bright")
generic_folder = os.path.join(mel_output_folder, "generic")


os.makedirs(soft_folder, exist_ok=True)
os.makedirs(bright_folder, exist_ok=True)
os.makedirs(generic_folder, exist_ok=True)


def save_mel_spectrogram(audio_file, output_path):
    y, sr = librosa.load(audio_file, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=224)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    
    plt.imsave(output_path, mel_spectrogram_db, cmap='viridis', dpi=224)
    img = Image.open(output_path).resize((224, 224))
    img.save(output_path)


label_mapping = {'bright': bright_folder, 'soft': soft_folder, 'generic': generic_folder}


for label in ['bright', 'soft', 'generic']:
    source_folder = os.path.join(source_audio_folder, label)
    count = 0

    if os.path.exists(source_folder):
        for audio_file in os.listdir(source_folder):
            if audio_file.endswith(".wav"):

                
                if count >= 240:
                    break

                audio_path = os.path.join(source_folder, audio_file)
                filename = os.path.splitext(audio_file)[0]

                
                mel_output_path = os.path.join(label_mapping[label], f"{filename}_mel.png")
                save_mel_spectrogram(audio_path, mel_output_path)

                count += 1
                print(f"Processed {audio_file}: Mel-spectrogram saved ({count}/{900}).")

print("Processing completed. Mel-spectrograms saved as 224x224 PNG images.")
