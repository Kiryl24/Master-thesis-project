import tensorflow as tf


physical_devices = tf.config.list_physical_devices('GPU')

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


if physical_devices:
    print("Znaleziono GPU:")
    for device in physical_devices:
        print(device)
    
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("Nie znaleziono GPU. Używany jest CPU.")
