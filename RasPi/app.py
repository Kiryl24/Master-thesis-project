import os
import time
import threading
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
from PIL import Image as PilImage, ImageTk
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

# Utwórz folder temp
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

    path = os.path.join("temp", "mel_spec_96x96.png")
    plt.figure(figsize=(8.5, 8.5), dpi=128)
    plt.axis('off')
    plt.imshow(mel_spec_db, cmap='viridis', aspect='auto', origin='lower')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return path

class SpectrogramApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spectrogram Classifier")
        self.model_path = "/home/kiryl/Documents/GitHub/Master-thesis-project/AI/Models/TM/model_96_TM.tflite"

        self.left_frame = tk.Frame(root, width=300)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.spectrogram_label = tk.Label(self.left_frame)
        self.spectrogram_label.pack()

        self.right_frame = tk.Frame(root)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.model_buttons = tk.Frame(self.right_frame)
        self.model_buttons.pack(pady=10)

        tk.Button(self.model_buttons, text="Digital", command=self.set_model_digital).pack(side=tk.LEFT)
        tk.Button(self.model_buttons, text="Acoustic", command=self.set_model_acoustic).pack(side=tk.LEFT)

        tk.Button(self.right_frame, text="Analyze", command=self.run_ai_script).pack(pady=10)
        tk.Button(self.right_frame, text="Help", command=self.show_help_video).pack(pady=5)
        tk.Button(self.right_frame, text="Shutdown", command=self.shutdown, fg="white", bg="red").pack(pady=5)

        self.log_label = tk.Label(self.right_frame, text="", wraplength=200, justify=tk.LEFT)
        self.log_label.pack(pady=20)

    def set_model_digital(self):
        self.model_path = "/home/kiryl/Documents/GitHub/Master-thesis-project/AI/Models/TM/model_96_TM.tflite"
        self.update_logs("Model set to: Digital")

    def set_model_acoustic(self):
        self.model_path = "/home/kiryl/Documents/GitHub/Master-thesis-project/AI/Models/TM/model_96_TM_acu.tflite"
        self.update_logs("Model set to: Acoustic")

    def run_ai_script(self):
        self.cleanup_temp_files()
        threading.Thread(target=self.process_audio).start()

    def shutdown(self):
        os.system("sudo shutdown -h now")

    def update_logs(self, text):
        self.log_label.config(text=text)

    def update_spectrogram(self, image_path):
        img = PilImage.open(image_path)
        img.thumbnail((300, 300))
        self.tk_img = ImageTk.PhotoImage(img)
        self.spectrogram_label.config(image=self.tk_img)

    def process_audio(self):
        try:
            for i in range(3, 0, -1):
                self.update_logs(f"Recording starts in {i}...")
                time.sleep(1)

            self.update_logs("Recording...")
            audio_data = record_audio()
            sf.write("temp/temp.wav", audio_data, 44100)

            self.update_logs("Generating spectrogram...")
            image_path = create_mel_spectrogram(audio_data)

            self.root.after(0, lambda: self.update_spectrogram(image_path))
            self.update_logs("Predicting label...")

            mel_image = PilImage.open(image_path)
            class_name, confidence_score = self.predict_label(mel_image)

            self.update_logs(f"Color: {class_name}\nConfidence: {confidence_score * 100:.2f}%")

        except Exception as e:
            self.update_logs(f"Error: {e}")

    def predict_label(self, mel_image):
        interpreter = tf.lite.Interpreter(model_path=self.model_path)
        interpreter.allocate_tensors()

        with open("labels.txt", "r") as f:
            class_names = f.readlines()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        mel_image = mel_image.convert('L')
        mel_image = mel_image.resize((96, 96))
        image_array = np.asarray(mel_image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        normalized_image_array = np.expand_dims(normalized_image_array, axis=-1)

        data = np.ndarray(shape=(1, 96, 96, 1), dtype=np.float32)
        data[0] = normalized_image_array

        interpreter.set_tensor(input_details[0]['index'], data)
        interpreter.invoke()

        prediction = interpreter.get_tensor(output_details[0]['index'])
        index = np.argmax(prediction)
        class_name = class_names[index].strip()[1:]
        confidence_score = prediction[0][index]
        return class_name, confidence_score

    def cleanup_temp_files(self):
        files = ["temp/temp.wav", "temp/mel_spec_96x96.png"]
        for file in files:
            if os.path.exists(file):
                os.remove(file)

    def show_help_video(self):
        messagebox.showinfo("Help", "Play the 'PianoInstruction.gif' manually.\nAnimated GIFs aren't supported natively in tkinter.")

# Uruchom aplikację
if __name__ == "__main__":
    root = tk.Tk()
    app = SpectrogramApp(root)
    root.geometry("800x400")
    root.mainloop()