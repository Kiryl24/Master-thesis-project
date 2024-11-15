import tensorflow as tf
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image


interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


class_names = ['bright', 'dark', 'generic']

def extract_mel_spectrogram(file_path, duration=4.0, sr=22050):
    audio, _ = librosa.load(file_path, sr=sr, duration=duration)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_img = Image.fromarray(mel_spec_db).resize((128, 128))
    mel_spec_img = np.array(mel_spec_img, dtype=np.float32) / 255.0
    mel_spec_img = mel_spec_img[..., np.newaxis]  
    return mel_spec_img

def predict_label(mel_spec):
    mel_spec = np.expand_dims(mel_spec, axis=0)  
    interpreter.set_tensor(input_details[0]['index'], mel_spec)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output_data)
    predicted_label = class_names[predicted_index]
    return predicted_label


file_path = 'WAV/bright/keyboard_acoustic_001-024-025.wav'


mel_spec_img = extract_mel_spectrogram(file_path)


predicted_label = predict_label(mel_spec_img)
print("Predicted Label:", predicted_label)


plt.figure(figsize=(5, 5))
plt.imshow(mel_spec_img.squeeze(), cmap='gray')
plt.title('MEL Spectrogram (128x128)')
plt.axis('off')
plt.show()
