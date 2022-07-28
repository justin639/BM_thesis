import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pylab as plt
import scipy.io
from customMobileNetV2 import customMobileNetV2 as cmv2


##############################################################################
# Show result
def show_result(history):
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


def getdata_from_mat(path, img_size):
    path = path + "/"

    mat_file_data = "M_AllFeatureKalman_Data_128.mat"
    mat_file = scipy.io.loadmat(path + mat_file_data)
    data_value = mat_file[mat_file_data[:-4]]

    IMG_SIZE = img_size
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

    return test_x, test_y, train_x, train_y


def create_model(img_size, momentum, classes, base_learning_rate):
    img_shape = (img_size, img_size, 3)
    # Create with custom MNV2
    base_model = cmv2(input_shape=img_shape,
                      momentum=momentum)

    base_model.trainable = True
    # Print base model architecture
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
                  loss=tf.keras.losses.CategoricalCrossentropy,
                  metrics=['accuracy'])
    return model


##############################################################################
# Todo extract the model build and calculate hyper-parameters
data_path = "NearFallPaper_DataCode"
batch_size = 16
img_size = 32
momentum = 0.9
# epoch마다 점점 줄여보기
base_learning_rate = 0.00001
validation_steps = 20
initial_epochs = 100
classes = 4
# Get Data from mat file
train_x, train_y, test_x, test_y = getdata_from_mat(data_path, img_size)

# Create model
model = create_model(img_size=img_size, momentum=momentum, classes=classes, base_learning_rate=base_learning_rate)
# Print model architecture
model.summary()

loss0, accuracy0 = model.evaluate(test_x, test_y, batch_size=batch_size, steps=validation_steps)
print("\nInitial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

# Train for epochs
history = model.fit(train_x, train_y,
                    epochs=initial_epochs,
                    validation_data=(test_x, test_y),
                    verbose=2)
# todo save weight as .h5 file
model_path = 'saved_models/customMNV2.h5'
# Save model
model.save(model_path)
# Reload model
new_model = keras.models.load_model(model_path)
new_model.summary()
# Check accuracy
loss, acc = new_model.evaluate(test_x,  test_y, verbose=2)
print('복원된 모델의 정확도: {:5.2f}%'.format(100*acc))

# Show weight result
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
# plt.figure(figsize=(initial_epochs, 3))
# plt.suptitle("Kernels of customMNV_2")
# for i in range(n_kernel):
#     f = kernel[:, :, :, i]
#     plt.subplot(3, n_kernel, i + 1)
#     plt.imshow(f[:, :, 0], cmap='gray')
#     plt.xticks([])
#     plt.yticks([])
#     plt.title(str(i))
# plt.show()

show_result(history)
