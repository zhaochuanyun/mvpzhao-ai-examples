import os
import h5py
import numpy as np
import keras
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt

base_dir = os.path.expanduser("~/tools/data/german/")

path_training = os.path.join(base_dir, 'validation.h5')
path_validation = os.path.join(base_dir, 'validation.h5')
path_test = os.path.join(base_dir, 'round1_test_a_20181109.h5')

fid_training = h5py.File(path_training, 'r')
fid_validation = h5py.File(path_validation, 'r')
fid_test = h5py.File(path_test, 'r')

print(list(fid_training.keys()))
print(list(fid_validation.keys()))
print(list(fid_test.keys()))

print("-" * 60)
print("training part")
s1_training = fid_training['sen1']
print(s1_training.shape)
s2_training = fid_training['sen2']
print(s2_training.shape)
label_training = fid_training['label']
print(label_training.shape)

print("-" * 60)
print("validation part")
s1_validation = fid_validation['sen1']
print(s1_validation.shape)
s2_validation = fid_validation['sen2']
print(s2_validation.shape)
label_validation = fid_validation['label']
print(label_validation.shape)

print("-" * 60)
print("test part")
s1_test = fid_test['sen1']
print(s1_test.shape)
s2_test = fid_test['sen2']
print(s2_test.shape)

s1_training = np.asarray(s1_training)
s1_training = s1_training.reshape(s1_training.shape[0], -1)

s2_training = np.asarray(s2_training)
s2_training = s2_training.reshape(s2_training.shape[0], -1)

data_training = np.hstack((s1_training, s2_training))
label_training = np.asarray(label_training)

print(data_training.shape)
print(label_training.shape)

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(18432,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(17, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

history = model.fit(data_training, label_training, epochs=20, batch_size=128, validation_split=0.1)

acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
