import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
import scipy.io
from customMobileNetV2 import customMobileNetV2 as cmv2

path = "NearFallPaper_Data_Code/"

mat_file_data = "M_AllFeatureKalman_Data_128.mat"
mat_file = scipy.io.loadmat(path + mat_file_data)
data_value = mat_file[mat_file_data[:-4]]

IMG_SIZE = 32
# print(mat_file_value[0][0][0][10379])

print("Number of data : " + str(data_value[0][0].size))
print("Input data shape : " + str(data_value[:, :, [0], [0]].shape))

train_x = np.zeros((data_value[0][0].size, IMG_SIZE, IMG_SIZE, 3))
print(train_x.shape)

for i in range(0, data_value[0][0].size):
    target = data_value[:, :, [0], [i]]
    W, L, H = target.shape
    image = np.zeros((W, L, 3))
    # print(image[:, :, 0].shape)
    image[:, :, 0] = target[..., 0]
    image[:, :, 1] = target[..., 0]
    image[:, :, 2] = target[..., 0]
    # resize input size with 3 different filters and add to train_x array
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    train_x[i] = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    # print("[" + str(i) + "] : " + str(train_x[i]))

test_x = train_x[0:500]
print(test_x.shape)

mat_file_label = "M_Label_One.mat"
mat_file = scipy.io.loadmat(path + mat_file_label)
train_y = mat_file[mat_file_label[:-4]]
train_y = tf.keras.utils.to_categorical(train_y)
print(train_y.shape)

test_y = train_y[0:500]
print(test_y.shape)

##############################################################################
# Todo extract the model build and calculate hyper-parameters
batch_size = 16
img_size = 32
momentum = 0.9
# epoch마다 점점 줄여보기
base_learning_rate = 0.00001
validation_steps = 20
initial_epochs = 100
classes = 4


IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create with custom MNV2
base_model = cmv2(input_shape=IMG_SHAPE,
                  momentum=momentum)

base_model.trainable = True
base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

# for 2 or more classes
prediction_layer = tf.keras.layers.Dense(classes, activation='softmax',
                                         use_bias=True, name='Logits')

model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

# for 2 or more classes
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

loss0, accuracy0 = model.evaluate(test_x, test_y, batch_size=batch_size, steps=validation_steps)
print("\ninitial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(train_x,train_y,
                    epochs=initial_epochs,
                    validation_data=(test_x, test_y),
                    verbose=2)
# todo save weight as .h5 file

# for layer in model.layers:
#     if 'conv' in layer.name:
#         kernel, biases = layer.get_weights()
#         print(layer.name, kernel.shape)  # 커널의 텐서 모양을 출력
#
# kernel, biases = model.layers[0].get_weights()  # 층 0의 커널 정보를 저장
# minv, maxv = kernel.min(), kernel.max()
# kernel = (kernel - minv) / (maxv - minv)
# n_kernel = 32
#
# plt.figure(figsize=(20, 3))
# plt.suptitle("Kernels of customMNV_2")
# for i in range(n_kernel):
#     f = kernel[:, :, :, i]
#     plt.subplot(3, n_kernel, i + 1)
#     plt.imshow(f[:, :, 0], cmap='gray')
#     plt.xticks([])
#     plt.yticks([])
#     plt.title(str(i))
# plt.show()

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
