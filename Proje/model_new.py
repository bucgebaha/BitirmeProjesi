import os, shutil

base_dir = 'C:\\Users\\bhabu\\Desktop\\Bitirme\\Proje\\DatasetCropped'
#os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
#os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
#os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
#os.mkdir(test_dir)

train_real_dir = os.path.join(train_dir, 'real')
#os.mkdir(train_dir)

train_fake_dir = os.path.join(train_dir, 'fake')
#os.mkdir(train_dir)

validation_real_dir = os.path.join(validation_dir, 'real')
#os.mkdir(validation_dir)

validation_fake_dir = os.path.join(validation_dir, 'fake')
#os.mkdir(validation_dir)

test_real_dir = os.path.join(test_dir, 'real')
#os.mkdir(test_dir)

test_fake_dir = os.path.join(test_dir, 'fake')
#os.mkdir(test_dir)  
    
from tensorflow.keras import layers
from tensorflow.keras import models

model = models.Sequential()
model.add(layers.Conv2D(8, (3, 3), activation='relu',
                        input_shape=(75, 75, 3)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(8, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

from tensorflow.keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(),
              metrics=['acc'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        # All images will be resized to 75x75
        target_size=(75, 75),
        batch_size=20,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(75, 75),
        batch_size=20,
        class_mode='binary')

history = model.fit(
      train_generator,
      steps_per_epoch=100,
      epochs=50,
      validation_data=validation_generator,
      validation_steps=50)

model.save('deepfake_detect_last.h5')

# model = models.load_model('deepfake_detect_last.h5')

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

import tensorflow as tf
import numpy as np
img_path = "C:\\Users\\bhabu\\Desktop\\real3.jpg"
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(75, 75))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
img_normalised = img_batch / 255
print(model.predict(img_normalised))

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='C:\\Users\\bhabu\\Desktop\\model_plot.png', show_shapes=True, show_layer_names=True)
