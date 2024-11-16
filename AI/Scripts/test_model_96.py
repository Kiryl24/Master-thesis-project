import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np


np.set_printoptions(suppress=True)


interpreter = tf.lite.Interpreter(model_path="AI/Models/TM/model_96_TM.tflite")
interpreter.allocate_tensors()


class_names = open("AI/Models/TM/labels.txt", "r").readlines()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


data = np.ndarray(shape=(1, 96, 96, 1), dtype=np.float32)


image = Image.open("Train_data/96Mels/generic/sample_27_mel.png")


image = image.convert("L")


size = (96, 96)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)


image_array = np.asarray(image)


normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1


normalized_image_array = np.expand_dims(normalized_image_array, axis=-1)


data[0] = normalized_image_array


input_index = input_details[0]['index']
interpreter.set_tensor(input_index, data)


interpreter.invoke()


output_index = output_details[0]['index']
prediction = interpreter.get_tensor(output_index)


index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]


print("Class:", class_name[2:], end="")  
print("Confidence Score:", confidence_score)
