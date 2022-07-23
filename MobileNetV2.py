import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from customMobileNetV2 import customMobileNetV2 as cmv2

# Data downloaded
builder = tfds.folder_dataset.ImageFolder('images/')
print(builder.info)
raw_train = builder.as_dataset(split='train', shuffle_files=True)
raw_test = builder.as_dataset(split='test', shuffle_files=True)
raw_valid = builder.as_dataset(split='valid', shuffle_files=True)

tfds.show_examples(raw_train, builder.info)

# Todo extract the model build and calculate hyper-parameters
batch_size = 16
img_size = 32
momentum = 0.9
# classes = 5
# epoch마다 점점 줄여보기
base_learning_rate = 0.00001
validation_steps = 20
initial_epochs = 100

# Our size is 32
IMG_SIZE = img_size


# Formatting data
def format_example(pair):
    image, label = pair['image'], pair['label']
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


train = raw_train.map(format_example)
validation = raw_valid.map(format_example)
test = raw_test.map(format_example)

BATCH_SIZE = batch_size
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

# what is this???
for image_batch, label_batch in train_batches.take(1):
    pass

print(image_batch.shape)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights=None)
# Create with custom MNV2
# base_model = cmv2(input_shape=IMG_SHAPE,
#                   momentum=momentum)

feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.trainable = True
# Set momentum of Batch Normalization Layer to 0.9 -> inner implemented
for layer in base_model.layers:
    if type(layer) == type(tf.keras.layers.BatchNormalization()):
        layer.momentum = momentum
base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)

prediction_layer = tf.keras.layers.Dense(1)
# for 2 or more classes
# prediction_layer = tf.keras.layers.Dense(classes, activation='softmax',
#                                          use_bias=True, name='Logits')
prediction_batch = prediction_layer(feature_batch_average)

model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
# for 2 or more classes
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
#               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
model.summary()

loss0, accuracy0 = model.evaluate(validation_batches, steps=validation_steps)
print("\ninitial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches,
                    verbose=2)
# todo save weight as .h5 file

# Show result
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
