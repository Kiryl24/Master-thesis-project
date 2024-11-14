import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
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


mel_image_shape = (96, 96, 1)
num_classes = 3


mel_input = Input(shape=mel_image_shape, name="mel_input")
x = Conv2D(16, (3, 3), activation='relu', padding='same')(mel_input)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax', name="output")(x)

model = Model(inputs=mel_input, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


mel_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
mel_train_dir = 'Train_data/MEL'

mel_generator = mel_datagen.flow_from_directory(
    mel_train_dir,
    target_size=(96, 96),
    color_mode='grayscale',
    batch_size=8,
    class_mode='sparse',
    shuffle=True
)


dataset = tf.data.Dataset.from_generator(
    lambda: mel_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 96, 96, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )
)


history = model.fit(
    dataset,
    steps_per_epoch=len(mel_generator),
    epochs=100,
    validation_data=dataset,
    validation_steps=len(mel_generator)
)


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_plot_E100.png')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot_E100.png')
plt.show()


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model_E100.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model TFLite saved as 'model_E100.tflite'")
