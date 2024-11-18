import os
import time
import numpy as np
import sounddevice as sd
import librosa
import soundfile as sf  
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import img_to_array

if not os.path.exists('temp'):
    os.makedirs('temp')

interpreter = tf.lite.Interpreter(model_path="model_96_TM.tflite")
interpreter.allocate_tensors()

class_names = open("labels.txt", "r").readlines()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

data = np.ndarray(shape=(1, 96, 96, 1), dtype=np.float32)


def record_audio(duration=4, sample_rate=22050):
    print("Recording for 4 seconds...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return audio_data.flatten()


def create_mel_spectrogram(audio, sample_rate=22050):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def save_spectrogram(image, filename):
    plt.figure(figsize=(1.28, 1.28), dpi=100)
    plt.axis('off')
    plt.imshow(image, cmap='gray', aspect='auto')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def predict_label(mel_image):
    mel_image = mel_image.convert("L").resize((96, 96))
    mel_array = img_to_array(mel_image).astype('float32') / 127.5 - 1

    mel_array = np.expand_dims(mel_array, axis=0)
    interpreter.set_tensor(input_details[0]['index'], mel_array)
    interpreter.invoke()

    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = np.argmax(predictions)
    confidence_score = predictions[0][predicted_label]

    class_name = class_names[predicted_label].strip()
    return class_name, confidence_score


while True:
    print("Starting recording...")

    for i in range(3, 0, -1):
        print(i)
        time.sleep(1)

    audio_data = record_audio()

    temp_wav_path = "temp/temp.wav"
    
    sf.write(temp_wav_path, audio_data, 22050)

    mel_spec = create_mel_spectrogram(audio_data)

    temp_img_path = "temp/mel_spec.png"
    mel_image = Image.fromarray(mel_spec)
    mel_image = mel_image.convert("L").resize((96, 96))
    save_spectrogram(mel_image, temp_img_path)

    class_name, confidence_score = predict_label(mel_image)
    print("Predicted class:", class_name)
    print("Confidence Score:", confidence_score)

    os.remove(temp_wav_path)
    os.remove(temp_img_path)

    print("Ready for the next recording...\n")
