import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pylab as plt
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import scipy.io
from tensorflow_core.python.keras.utils import np_utils
import seaborn as sns
from customMobileNetV2 import customMobileNetV2 as cmv2
from tensorflow.keras.applications import MobileNetV2


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
    plt.savefig("results.png")
    # plt.show()


def show_confusion_matrix(model, x_test, y_test):
    class_names = [1, 2, 3, 4, 5, 6, 7]  # name  of classes

    pred = model.predict(x_test)
    pred = np_utils.to_categorical(np.argmax(pred, axis=1))
    conf_mat = confusion_matrix(y_test.argmax(axis=1), pred.argmax(axis=1))
    print("Confussion Matrix")
    print(conf_mat)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(pd.DataFrame(conf_mat), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Resnet Confusion matrix Opti')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    tick_marks = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    plt.savefig("confusion_matrix.png")


def getBMData(path, model_type='CNN'):
    CNNXdata_file_name = path + "/xdata.npy"
    NNXdata_file_name = path + "/xdata2.npy"
    Ydata_file_name = path + "/ydata.npy"

    CNNXdata = np.load(CNNXdata_file_name, allow_pickle=True)  # 데이터 로드. @파일명
    NNXdata = np.load(NNXdata_file_name)  # 데이터 로드. @파일명
    Ydata = np.load(Ydata_file_name)  # 데이터 로드. @파일명

    if CNNXdata.ndim == 3:
        CNNXdata = CNNXdata.reshape(CNNXdata.shape[0], CNNXdata.shape[1], CNNXdata.shape[2], 1)
    else:
        CNNXdata = CNNXdata

    if NNXdata.ndim == 3:
        NNXdata = NNXdata.reshape(NNXdata.shape[0], (NNXdata.shape[1] * NNXdata.shape[2]))
    else:
        NNXdata = NNXdata

    if Ydata.ndim == 3:
        CNNYdata = Ydata.reshape(Ydata.shape[0], (Ydata.shape[1] * Ydata.shape[2]))
        NNYdata = Ydata.reshape(Ydata.shape[0])
    else:
        Ydata = Ydata

    CNNXdata = expand_3D(CNNXdata)
    CNNYdata = tf.keras.utils.to_categorical(CNNYdata)

    if model_type == 'CNN':
        return train_test_split(CNNXdata, CNNYdata, test_size=0.2, random_state=1)
    if model_type == 'ANN':
        return train_test_split(NNXdata, NNYdata, test_size=0.2, random_state=1)


def expand_3D(data_value):
    num, w, l, c = data_value.shape
    train_x = np.zeros((num, w, l, 3))

    for i in range(0, num):
        target = data_value[i, ...]
        W, L, _ = target.shape
        image = np.zeros((W, L, 3))
        image[:, :, 0] = target[0, ...]
        image[:, :, 1] = target[0, ...]
        image[:, :, 2] = target[0, ...]
        # resize input size with 3 different filters and add to train_x array
        image = tf.cast(image, tf.float32)
        train_x[i] = image
    return train_x


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


def create_model(base_learning_rate, img_size=35, momentum=0.9, classes=7, model_type='custom'):
    img_shape = (img_size, img_size, 3)
    # Create with custom MNV2
    if model_type == 'custom':
        base_model = cmv2(input_shape=img_shape,
                          momentum=momentum)

    if model_type == 'tensorflow':
        base_model = MobileNetV2(input_shape=img_shape, include_top=False, weights=None)
        # Set momentum of Batch Normalization Layer to 0.9 -> inner implemented
        for layer in base_model.layers:
            if type(layer) == type(tf.keras.layers.BatchNormalization()):
                layer.momentum = momentum

    base_model.trainable = True
    # Print base model architecture
    # base_model.summary()

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

    # plot graph of model
    plot_model(model, to_file='MNV2model.png', show_shapes=True)

    return model

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

##############################################################################
