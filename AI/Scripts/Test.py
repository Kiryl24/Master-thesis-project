import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import tensorflow as tf
from PIL import Image
import os




def save_mel_spectrogram(audio_file, output_path):
    y, sr = librosa.load(audio_file, sr=None)

    
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

    
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    
    mel_spectrogram_resized = np.resize(mel_spectrogram_db, (128, 128))

    
    mel_image = Image.fromarray(np.uint8((mel_spectrogram_resized - np.min(mel_spectrogram_resized)) * 255 / (
                np.max(mel_spectrogram_resized) - np.min(mel_spectrogram_resized))))
    mel_image.save(output_path)



def save_chromagram(audio_file, output_path):
    y, sr = librosa.load(audio_file, sr=None)

    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    
    chroma_normalized = librosa.util.normalize(chroma)

    
    chroma_resized = np.resize(chroma_normalized, (128, 128))

    
    chroma_image = Image.fromarray(np.uint8(chroma_resized * 255))  
    chroma_image.save(output_path)



audio_file = 'Acoustic Grand MS4 - MuseSounds.wav'  
mel_output_path = 'output_mel_spectrogram.png'  
chroma_output_path = 'output_chromagram.png'  

save_mel_spectrogram(audio_file, mel_output_path)
save_chromagram(audio_file, chroma_output_path)

print("Mel-spectrogram i chromagram zostały zapisane.")


def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter



def preprocess_image(image_path, image_size=(128, 128)):
    
    image = Image.open(image_path).convert('L')  
    image = image.resize(image_size)  

    
    image_array = np.array(image, dtype=np.float32)

    
    image_array = np.expand_dims(image_array, axis=-1)  
    image_array = np.expand_dims(image_array, axis=0)  
    image_array /= 255.0  

    return image_array



def analyze_with_model(model, mel_image_path, chroma_image_path):
    
    mel_image = preprocess_image(mel_image_path)
    chroma_image = preprocess_image(chroma_image_path)

    
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    
    model.set_tensor(input_details[0]['index'], mel_image)
    model.invoke()
    mel_output = model.get_tensor(output_details[0]['index'])

    
    model.set_tensor(input_details[0]['index'], chroma_image)
    model.invoke()
    chroma_output = model.get_tensor(output_details[0]['index'])

    
    return mel_output, chroma_output



if __name__ == "__main__":
    
    model_path = 'path_to_your_model.tflite'  
    model = load_tflite_model(model_path)

    
    mel_image_path = 'output_mel_spectrogram.png'  
    chroma_image_path = 'output_chromagram.png'  

    
    mel_output, chroma_output = analyze_with_model(model, mel_image_path, chroma_image_path)

    
    print("Mel-Spectrogram Output:", mel_output)
    print("Chromagram Output:", chroma_output)
