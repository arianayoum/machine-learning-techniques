# -*- coding: utf-8 -*-
"""
**Load MNIST digits**
"""

from keras.datasets import mnist
(train_X, train_y), (test_X, test_y) = mnist.load_data()

"""**Show some of the hand-written images**


"""

import matplotlib.pyplot as plt

train_images = train_X / 255.0
test_images = test_X / 255.0

plt.figure(figsize=(10, 10))

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.subplots_adjust(hspace=.3)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.title(train_y[i])
plt.show()

"""Explore the MNIST handwritten digits dataset.

"""

print(train_y.shape)
print(train_images.shape)
print(test_images.shape)

import tensorflow as tf

X_train = train_images.reshape((train_images.shape[0], 28, 28, 1))
X_test = test_images.reshape((test_images.shape[0], 28, 28, 1))

print(X_train.shape)

tf.random.set_seed(42)

"""Design a shallow neural network classifier with at most 3 layers (plus the final softmax layer) and try to classify the digits."""

from tensorflow.keras import datasets, layers, models, losses
model_3layers = models.Sequential()
model_3layers.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model_3layers.add(layers.MaxPooling2D((2, 2)))
model_3layers.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model_3layers.add(layers.MaxPooling2D((2, 2)))
model_3layers.add(layers.Flatten())
model_3layers.add(layers.Dense(10, activation='softmax'))

model_3layers.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_3layers.summary()

model_3layers.fit(X_train, train_y, validation_data=(X_test, test_y), epochs=1)

test_loss, test_acc = model_3layers.evaluate(X_test, test_y, verbose=2)

print('Accuracy on test set:', test_acc)

predictions = model_3layers.predict(X_test)
print('Predictions for the first image:')
print(predictions[0])

"""**Report your network design and accuracy.**
# This three-layered model had an accuracy of 97.2% when validated on the test set. This isn't bad, but the best neural network models have an accuracy of 99%. This suggests that we may want to add more layers to the network to enhance accuracy.

Apply data augmentation. How does test accuracy change?
"""

from warnings import catch_warnings
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(height_shift_range=10,
                             width_shift_range=10,
                             horizontal_flip=True,
                             vertical_flip=True
                             )

model3_aug = tf.keras.models.clone_model(model_3layers)

model3_aug.compile(optimizer='adam',
              loss=losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

train_generator = datagen.flow(X_train, train_y, seed=42, batch_size=40)
model3_aug.fit(train_generator, epochs=100, validation_data=(X_test, test_y))

test_loss, test_acc = model3_aug.evaluate(X_test, test_y, verbose=2)
print('Accuracy on test set:', test_acc)

"""# After augmentation, the accuracy of the 3-layer model decreased to 91.3%. This decrease is due to the fact that the model is now prepared to deal with variations in images (e.g. horizontally flipped images), which did not necessarily appear in the test set. All in all, although the accuracy slightly decreased, our model has now become more robust.

Design a deep network with at least 9 layers and try to classify the digits.
"""

from tensorflow.keras import datasets, layers, models, losses
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss=losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

model.summary()

model.fit(X_train, train_y, validation_data=(X_test, test_y), epochs=1)

test_loss, test_acc = model.evaluate(X_test, test_y, verbose=2)

print('Accuracy on test set:', test_acc)

predictions = model.predict(X_test)
print('Predictions for the first image:')
print(predictions[0])

"""**Report your network design and accuracy.**

# Before I comment on the current 9-layered model, I was playing around with different numbers of layers and I saw that a 5-layered model produced an accuracy rate of 98.7%, which is pretty good. However, my 9-layered model only had an accuracy of 97.9%, suggesting that there is an issue of over-fitting that is taking place in my model.

Apply data augmentation. How does test accuracy change?
"""

from warnings import catch_warnings
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(height_shift_range=10,
                             width_shift_range=10,
                             horizontal_flip=True,
                             vertical_flip=True
                             )

model_aug = tf.keras.models.clone_model(model)

model_aug.compile(optimizer='adam',
              loss=losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

train_generator = datagen.flow(X_train, train_y, seed=42, batch_size=40)
model_aug.fit(train_generator, epochs=100, validation_data=(X_test, test_y))

test_loss, test_acc = model_aug.evaluate(X_test, test_y, verbose=2)
print('Accuracy on test set:', test_acc)

"""**Discuss your results.**
# After augmentation, the accuracy slightly decreased but not drastically so. In fact, considering that it has now become more robust, compared to the ~91% accuracy of the augmented 3-layer model, this augmented 9-layer model seems to be perfoming quite well. However, it may suffer from over-fitting (as mentioned in the previous description of the model before augmentation) so it may be worth removing some layers.
"""
