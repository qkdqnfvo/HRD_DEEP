# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.utils import to_categorical
# from tensorflow import keras


# (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# Y_train = to_categorical(Y_train)
# Y_test = to_categorical(Y_test)
# model1 = keras.models.Sequential(
#     [
#         keras.layers.Flatten(input_shape=(28, 28)),
#         keras.layers.Dense(256, activation='relu'),
#         keras.layers.Dense(128, activation='relu'),
#         keras.layers.Dense(10, activation='softmax')
#     ]
# )
# model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# history = model1.fit(X_train, Y_train, epochs=100, batch_size=128)
# print(model1.evaluate(X_test, Y_test))


import tensorflow as tf
from tensorflow.python.client import device_lib


print(device_lib.list_local_devices())

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

print(tf.test.gpu_device_name())
