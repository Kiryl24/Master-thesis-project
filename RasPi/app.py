import os
os.environ["KIVY_VIDEO"] = "ffpyplayer"
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.video import Video
from kivy.uix.popup import Popup
from threading import Thread
import time
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
from PIL import Image as PilImage, ImageOps
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from kivy.clock import Clock  

from kivy.config import Config
Config.set('graphics', 'width', '480')
Config.set('graphics', 'height', '320')
Config.set('graphics', 'borderless', '1')  
Config.set('graphics', 'resizable', '0')

if not os.path.exists('temp'):
    os.makedirs('temp')


interpreter = tf.lite.Interpreter(model_path="/home/kiryl/Documents/GitHub/Master-thesis-project/RasPi/model_96_TM.tflite")
interpreter.allocate_tensors()
class_names = open("labels.txt", "r").readlines()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def record_audio(duration=4, sample_rate=22050):
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return audio_data.flatten()

def create_mel_spectrogram(audio, sample_rate=22050):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128, fmax=8000)
    return librosa.power_to_db(mel_spec, ref=np.max)

def save_spectrogram(image, filename, colormap='viridis'):
    plt.figure(figsize=(1.28, 1.28), dpi=100)
    plt.axis('off')
    plt.imshow(image, cmap=colormap, aspect='auto')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_grayscale_spectrogram(image, filename, size=(96, 96)):
    grayscale_image = ImageOps.grayscale(PilImage.fromarray(image))
    grayscale_image = grayscale_image.resize(size)
    grayscale_image.save(filename)

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

class MainApp(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='horizontal', **kwargs)

        self.left_panel = BoxLayout(orientation='vertical', size_hint=(0.6, 1))
        self.spectrogram_image = Image()
        self.left_panel.add_widget(self.spectrogram_image)

        self.right_panel = BoxLayout(orientation='vertical', size_hint=(0.4, 1))
        self.button_analyze = Button(text="Analyze", size_hint=(1, 0.2))
        self.button_analyze.bind(on_press=self.run_ai_script)
        self.right_panel.add_widget(self.button_analyze)

        self.button_help = Button(text="Help", size_hint=(1, 0.2), background_color=(1, 0, 0, 1))
        self.button_help.bind(on_press=self.show_help_video)
        self.right_panel.add_widget(self.button_help)

        self.logs = Label(text="", size_hint=(1, 0.6))
        self.right_panel.add_widget(self.logs)

        self.add_widget(self.left_panel)
        self.add_widget(self.right_panel)

        self.play_intro_animation()

    def run_ai_script(self, instance):
        self.cleanup_temp_files()
        Thread(target=self.process_audio).start()

    def process_audio(self):
        try:
            for i in range(3, 0, -1):
                self.update_logs(f"Recording starts in {i}...")
                time.sleep(1)

            self.update_logs("Recording...")
            audio_data = record_audio()
            temp_wav_path = "temp/temp.wav"
            librosa.output.write_wav(temp_wav_path, audio_data, sr=22050)

            self.update_logs("Generating spectrogram...")
            mel_spec = create_mel_spectrogram(audio_data)
            temp_spectrogram_path = "temp/mel_spec_128x128.png"
            save_spectrogram(mel_spec, temp_spectrogram_path)

            temp_grayscale_path = "temp/mel_spec_96x96.png"
            save_grayscale_spectrogram(mel_spec, temp_grayscale_path)

            self.update_spectrogram(temp_spectrogram_path)

            self.update_logs("Predicting label...")
            mel_image = PilImage.fromarray(mel_spec)
            class_name, confidence_score = predict_label(mel_image)
            self.update_logs(f"Predicted: {class_name} ({confidence_score * 100:.2f}%)")
        except Exception as e:
            self.update_logs(f"Error: {str(e)}")

    def update_logs(self, text):
        self.logs.text = text

    def update_spectrogram(self, image_path):
        self.spectrogram_image.source = image_path
        self.spectrogram_image.reload()

    def cleanup_temp_files(self):
        temp_files = ["temp/temp.wav", "temp/mel_spec_128x128.png", "temp/mel_spec_96x96.png"]
        for file in temp_files:
            if os.path.exists(file):
                os.remove(file)

    def play_intro_animation(self):
        intro_video = Video(source="/home/kiryl/Documents/GitHub/Master-thesis-project/RasPi/intro.mp4", size_hint=(None, None), size=(480, 320), state='play')
        popup = Popup(title="Intro", content=intro_video, size_hint=(None, None), size=(480, 320))
        popup.open()
        intro_video.bind(on_stop=lambda instance: self.close_video(popup, intro_video))
        Clock.schedule_once(lambda dt: self.close_video(popup, intro_video), 3)  

    def show_help_video(self, instance):
        help_video = Video(source="/home/kiryl/Documents/GitHub/Master-thesis-project/RasPi/PianoInstruction.mp4", size_hint=(None, None), size=(480, 320), state='play')
        popup = Popup(title="Help", content=help_video, size_hint=(None, None), size=(480, 320))
        popup.open()
        help_video.bind(on_stop=lambda instance: self.close_video(popup, help_video))
        Clock.schedule_once(lambda dt: self.close_video(popup, help_video), 6)

    def close_video(self, popup, video):
        popup.dismiss()
        video.state = 'stop'
        video.unload()

class SpectrogramApp(App):
    def build(self):
        return MainApp()

if __name__ == "__main__":
    SpectrogramApp().run()
