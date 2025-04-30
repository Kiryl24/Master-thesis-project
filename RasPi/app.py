import os

from kivy.utils import get_color_from_hex
from kivy.graphics.texture import Texture
from kivy.uix.image import AsyncImage, Image
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label

from kivy.uix.popup import Popup
from threading import Thread
import time
from subprocess import call
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
from PIL import Image as PilImage, ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from kivy.clock import Clock
import soundfile as sf
from kivy.config import Config

Config.set('graphics', 'width', '480')
Config.set('graphics', 'height', '320')
Config.set('graphics', 'borderless', '1')
Config.set('graphics', 'resizable', '0')

if not os.path.exists('temp'):
    os.makedirs('temp')


def record_audio(duration=4, sample_rate=44100):
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    audio_data = audio_data / np.max(np.abs(audio_data))
    return audio_data.flatten()


def create_mel_spectrogram(audio, sample_rate=44100):
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

    mel_spec_path_128x128 = os.path.join("temp", "mel_spec_96x96.png")
    plt.figure(figsize=(8.5, 8.5), dpi=128)
    plt.axis('off')
    plt.imshow(mel_spec_db, cmap='viridis', aspect='auto', origin='lower')
    plt.savefig(mel_spec_path_128x128, bbox_inches='tight', pad_inches=0)
    plt.close()

    return mel_spec_path_128x128


class MainApp(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='horizontal', **kwargs)
        self.model_path = "model_96_TM.tflite"

        self.left_panel = BoxLayout(orientation='vertical', size_hint=(0.6, 1))
        self.spectrogram_image = AsyncImage()
        self.left_panel.add_widget(self.spectrogram_image)

        self.right_panel = BoxLayout(orientation='vertical', size_hint=(0.4, 1))

        self.model_selector_box = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        self.button_digital = Button(text="Digital", font_size=12)
        self.button_acoustic = Button(text="Acoustic", font_size=12)
        self.button_digital.bind(on_press=self.set_model_digital)
        self.button_acoustic.bind(on_press=self.set_model_acoustic)
        self.model_selector_box.add_widget(self.button_digital)
        self.model_selector_box.add_widget(self.button_acoustic)
        self.right_panel.add_widget(self.model_selector_box)

        self.button_analyze = Button(text="Analyze", size_hint=(1, 0.2),
                                     background_color=(get_color_from_hex('#ffffff')),
                                     color=(get_color_from_hex('#070707')))
        self.button_analyze.bind(on_press=self.run_ai_script)
        self.right_panel.add_widget(self.button_analyze)

        self.button_help = Button(text="Help", size_hint=(1, 0.2), color=(get_color_from_hex('#e76b00')),
                                  background_color=(get_color_from_hex('#ffffff')))
        self.button_help.bind(on_press=self.show_help_video)
        self.right_panel.add_widget(self.button_help)

        self.logs = Label(text="", size_hint=(1, 0.4))
        self.right_panel.add_widget(self.logs)

        self.button_off = Button(text="Shutdown", size_hint=(1, 0.15), background_color=(get_color_from_hex('#FF0000')))
        self.button_off.bind(on_press=self.off)
        self.right_panel.add_widget(self.button_off)

        self.add_widget(self.left_panel)
        self.add_widget(self.right_panel)

    def set_model_digital(self, instance):
        self.model_path = "model_96_TM.tflite"
        self.update_logs("Model set to: Digital")

    def set_model_acoustic(self, instance):
        self.model_path = "model_96_TM_acu.tflite"
        self.update_logs("Model set to: Acoustic")

    def run_ai_script(self, instance):
        self.cleanup_temp_files()
        Thread(target=self.process_audio).start()

    def off(self, instance):
        call("sudo shutdown -h now", shell=True)

    def process_audio(self):
        try:
            for i in range(3, 0, -1):
                Clock.schedule_once(lambda dt, i=i: self.update_logs(f"Recording starts in {i}..."))
                time.sleep(1)

            Clock.schedule_once(lambda dt: self.update_logs("Recording..."))
            audio_data = record_audio()
            temp_wav_path = "temp/temp.wav"
            sf.write(temp_wav_path, audio_data, 44100)

            Clock.schedule_once(lambda dt: self.update_logs("Generating spectrograms..."))
            mel_spec_path_128x128 = create_mel_spectrogram(audio_data)

            Clock.schedule_once(lambda dt: self.update_spectrogram(mel_spec_path_128x128))

            Clock.schedule_once(lambda dt: self.update_logs("Predicting label..."))
            mel_image = PilImage.open(mel_spec_path_128x128)

            class_name, confidence_score = self.predict_label(mel_image)
            Clock.schedule_once(lambda dt: self.update_logs(f"Color: {class_name}\n{confidence_score * 100:.2f}%"))

        except Exception as e:
            print(f"{e}")
            Clock.schedule_once(lambda dt: self.update_logs(f"Error processing audio:\n{e}"))

    def predict_label(self, mel_image):
        interpreter = tf.lite.Interpreter(model_path=self.model_path)
        interpreter.allocate_tensors()

        class_names = open("labels.txt", "r").readlines()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        mel_image = mel_image.convert('L')
        mel_image = mel_image.resize((96, 96))
        image_array = np.asarray(mel_image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        normalized_image_array = np.expand_dims(normalized_image_array, axis=-1)

        data = np.ndarray(shape=(1, 96, 96, 1), dtype=np.float32)
        data[0] = normalized_image_array

        input_index = input_details[0]['index']
        interpreter.set_tensor(input_index, data)
        interpreter.invoke()

        output_index = output_details[0]['index']
        prediction = interpreter.get_tensor(output_index)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()[1:]
        confidence_score = prediction[0][index]

        print(f"Sound: {class_name}")
        print(f"Confidence Score: {confidence_score}")
        return class_name, confidence_score

    def update_logs(self, text):
        self.logs.text = text

    def update_spectrogram(self, image_path):
        try:
            if os.path.exists(image_path):
                self.spectrogram_image.source = image_path
                self.spectrogram_image.reload()
                self.spectrogram_image.nocache = False
                print(f"Spectrogram updated with image from {image_path}")
            else:
                print(f"File not found: {image_path}")
        except Exception as e:
            print(f"Error updating spectrogram: {e}")

    def cleanup_temp_files(self):
        temp_files = ["temp/temp.wav", "temp/mel_spec_128x128.png", "temp/mel_spec_96x96.png"]
        for file in temp_files:
            if os.path.exists(file):
                os.remove(file)

    def show_intro_popup(self):
        gif_image = Image(
            source="intro.gif",
            anim_delay=0.05,
            size_hint=(None, None),
            size=(480, 320)
        )
        popup = Popup(
            title="Welcome!",
            content=gif_image,
            size_hint=(None, None),
            size=(480, 320)
        )
        popup.open()
        Clock.schedule_once(lambda dt: popup.dismiss(), 5)

    def show_help_video(self, instance):
        gif_image = Image(
            source="PianoInstruction.gif",
            anim_delay=0.05,
            size_hint=(None, None),
            size=(480, 320)
        )
        popup = Popup(
            title="Help",
            content=gif_image,
            size_hint=(None, None),
            size=(480, 320)
        )
        popup.open()
        Clock.schedule_once(lambda dt: popup.dismiss(), 4)


class SpectrogramApp(App):
    def build(self):
        app_layout = MainApp()
        Clock.schedule_once(lambda dt: app_layout.show_intro_popup(), 0.1)
        return app_layout


if __name__ == "__main__":
    SpectrogramApp().run()
