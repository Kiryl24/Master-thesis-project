import os
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, \
    Activation, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2  # Import regularizacji L2
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("Found GPUs:", physical_devices)
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("No GPU found. Using CPU.")

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',    # Monitorujemy walidacyjną stratę
    factor=0.5,            # Zmniejsz współczynnik uczenia o połowę, gdy metryka nie poprawia się
    patience=3,            # Czekaj 3 epoki bez poprawy, zanim zmniejszysz lr
    verbose=1,             # Wyświetl komunikaty, gdy zmienia się współczynnik uczenia
    min_lr=0.00001         # Minimalna wartość współczynnika uczenia
)
def create_deep_model(input_shape, num_classes):
    inputs = Input(shape=input_shape, name="mel_input")

    # Zastosowanie L2 regularizacji oraz zwiększenie Dropout do 0.6
    x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(0.01))(inputs)  # Zmieniono na Conv2D z L2 regularizacją
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(0.01))(x)  # Zmieniono na Conv2D z L2 regularizacją
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.6)(x)  # Zwiększono Dropout do 0.6

    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.01))(x)  # Zmieniono na Conv2D z L2 regularizacją
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.01))(x)  # Zmieniono na Conv2D z L2 regularizacją
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.6)(x)  # Zwiększono Dropout do 0.6

    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.01))(x)  # Zmieniono na Conv2D z L2 regularizacją
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.01))(x)  # Zmieniono na Conv2D z L2 regularizacją
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.6)(x)  # Zwiększono Dropout do 0.6

    shortcut = x
    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.01))(x)  # Zmieniono na Conv2D z L2 regularizacją
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.01))(x)  # Zmieniono na Conv2D z L2 regularizacją
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    shortcut = Conv2D(128, (1, 1), padding='same', kernel_regularizer=l2(0.01))(shortcut)  # Zmieniono na Conv2D z L2 regularizacją
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.6)(x)  # Zwiększono Dropout do 0.6

    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.6)(x)  # Zwiększono Dropout do 0.6
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.6)(x)  # Zwiększono Dropout do 0.6

    outputs = Dense(num_classes, activation='softmax', name="output")(x)

    model = Model(inputs, outputs)
    return model


mel_image_shape = (128, 128, 1)
num_classes = 3

model = create_deep_model(mel_image_shape, num_classes)

model.compile(optimizer=Adam(learning_rate=0.00003),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

mel_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)  # Poprawiona wartość skali
mel_train_dir = 'Train_data/128Mels'

mel_generator = mel_datagen.flow_from_directory(
    mel_train_dir,
    target_size=(128, 128),
    color_mode='rgb',
    batch_size=8,
    class_mode='sparse',
    shuffle=True
)

dataset = tf.data.Dataset.from_generator(
    lambda: mel_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 128, 128, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )
)

history = model.fit(
    dataset,
    steps_per_epoch=len(mel_generator),
    epochs=50,
    validation_data=dataset,
    validation_steps=len(mel_generator),
    callbacks=[reduce_lr]  # Dodajemy callback
)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.axhline(y=1, color='black', linestyle='--', label='y = 1')
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

with open('model_gray.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model TFLite saved as 'model.tflite'")
