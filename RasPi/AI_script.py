import os
import time
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


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
    mel_array = img_to_array(mel_image).astype('float32') / 255.0
    mel_array = np.expand_dims(mel_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], mel_array)
    interpreter.invoke()

    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = np.argmax(predictions)
    return predicted_label


while True:
    print("Starting recording...")

    for i in range(3, 0, -1):
        print(i)
        time.sleep(1)

    audio_data = record_audio()

    librosa.output.write_wav("temp.wav", audio_data, sr=22050)

    mel_spec = create_mel_spectrogram(audio_data)
    save_spectrogram(mel_spec, "mel_spec.png")

    mel_image = Image.open("mel_spec.png").convert("L").resize((128, 128))

    predicted_label = predict_label(mel_image)
    print("Predicted label:", predicted_label)

    os.remove("temp.wav")
    os.remove("mel_spec.png")

    print("Ready for the next recording...\n")
