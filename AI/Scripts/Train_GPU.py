import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Dropout, concatenate
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt


physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("Found GPUs:", physical_devices)
    
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("No GPU found. Using CPU.")

mel_image_shape = (128, 128, 1)
chroma_image_shape = (128, 128, 1)
num_classes = 3


mel_input = Input(shape=mel_image_shape, name="mel_input")
x = Conv2D(32, (3, 3), activation='relu')(mel_input)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = GlobalAveragePooling2D()(x)  
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
mel_output = Dense(64, activation='relu')(x)


chroma_input = Input(shape=chroma_image_shape, name="chroma_input")
y = Conv2D(32, (3, 3), activation='relu')(chroma_input)
y = Conv2D(64, (3, 3), activation='relu')(y)
y = GlobalAveragePooling2D()(y)  
y = Dense(64, activation='relu')(y)
y = Dropout(0.5)(y)
chroma_output = Dense(64, activation='relu')(y)


combined = concatenate([mel_output, chroma_output])
z = Dense(256, activation='relu')(combined)
z = Dropout(0.5)(z)
z = Dense(num_classes, activation='softmax', name="output")(z)


model = Model(inputs=(mel_input, chroma_input), outputs=z)
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


mel_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
chroma_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

mel_train_dir = 'Train_data/MEL'
chroma_train_dir = 'Train_data/Chromagrams'


mel_generator = mel_datagen.flow_from_directory(
    mel_train_dir,
    target_size=(128, 128),
    color_mode='grayscale',
    batch_size=8,
    class_mode='sparse',
    shuffle=True
)

chroma_generator = chroma_datagen.flow_from_directory(
    chroma_train_dir,
    target_size=(128, 128),
    color_mode='grayscale',
    batch_size=8,
    class_mode='sparse',
    shuffle=True
)


def combined_generator(mel_gen, chroma_gen):
    while True:
        mel_batch, mel_labels = next(mel_gen)
        chroma_batch, chroma_labels = next(chroma_gen)
        yield (mel_batch, chroma_batch), mel_labels


output_signature = (
    (
        tf.TensorSpec(shape=(None, 128, 128, 1), dtype=tf.float32),  
        tf.TensorSpec(shape=(None, 128, 128, 1), dtype=tf.float32)     
    ),
    tf.TensorSpec(shape=(None,), dtype=tf.int32)  
)

dataset = tf.data.Dataset.from_generator(
    lambda: combined_generator(mel_generator, chroma_generator),
    output_signature=output_signature
)


history = model.fit(
    dataset,
    steps_per_epoch=min(len(mel_generator), len(chroma_generator)),
    epochs=50,
    validation_data=dataset,
    validation_steps=min(len(mel_generator), len(chroma_generator))
)


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()


with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model TFLite saved as 'model.tflite'")
