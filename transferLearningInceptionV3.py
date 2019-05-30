import os
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

data_dir = 'data'
train_dir = os.path.join(data_dir, 'train')
validation_dir = os.path.join(data_dir, 'validation')

IMG_SIZE = 160
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

base_model = keras.applications.InceptionV3(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
base_model.summary()

layer_after_mixed9_index = None
for index, layer in enumerate(base_model.layers):
    if layer.name == 'mixed9':
        layer_after_mixed9_index = index + 1
        break

for layer in base_model.layers[:layer_after_mixed9_index]:
    layer.trainable = False

for layer in base_model.layers[layer_after_mixed9_index:]:
    layer.trainable = True

global_average_pooling_layer = layers.GlobalAveragePooling2D()
dense_layer = layers.Dense(1024, activation='relu')
prediction_layer = layers.Dense(1, activation='sigmoid')

model = keras.Sequential([
    base_model,
    global_average_pooling_layer,
    dense_layer,
    prediction_layer
])

model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1/255,
                                   rotation_range=15,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1/255)

BATCH_SIZE = 16

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='binary',
                                                    target_size=(IMG_SIZE, IMG_SIZE))

validation_generator = validation_datagen.flow_from_directory(train_dir,
                                                              batch_size=BATCH_SIZE,
                                                              class_mode='binary',
                                                              target_size=(IMG_SIZE, IMG_SIZE))

history = model.fit_generator(train_generator, epochs=15, validation_data=validation_generator)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

alien_image = image.load_img('data/alien.jpg', target_size=(IMG_SIZE, IMG_SIZE))
alien_image_array = image.img_to_array(alien_image) / 255
alien_image_array = np.expand_dims(alien_image_array, axis=0)

predator_image = image.load_img('data/predator.jpg', target_size=(IMG_SIZE, IMG_SIZE))
predator_image_array = image.img_to_array(predator_image) / 255
predator_image_array = np.expand_dims(predator_image_array, axis=0)

images = np.vstack([alien_image_array, predator_image_array])

result = model.predict(images)

for index, prediction in enumerate(result):
    label = 'alien' if prediction < 0.5 else 'predator'
    print('%d image is %s' % (index, label))
