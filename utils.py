import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pylab as plt
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, auc
from sklearn.model_selection import train_test_split
import scipy.io
from numpy import interp
from tensorflow_core.python.keras.utils import np_utils
import seaborn as sns
from itertools import cycle
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

    y_pred = model.predict(x_test)
    pred = np_utils.to_categorical(np.argmax(y_pred, axis=1))
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


def show_roc_curve(model, x_test, y_test):
    class_names = [1, 2, 3, 4, 5, 6, 7]  # name  of classes
    y_pred = model.predict(x_test)
    numclasses = len(class_names)
    # Plot linewidth.
    lw = 2
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(numclasses):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(numclasses)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(numclasses):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= numclasses

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average(area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average(area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(["red", "blue", "orange", "gold", "aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(numclasses), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("MobileNetV2 Roc curve")
    plt.legend(loc="lower right")
    plt.savefig("roc_graph.png")


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
        CNNYdata = Ydata

    CNNXdata = expand_3D(CNNXdata)
    CNNYdata = tf.keras.utils.to_categorical(CNNYdata)

    if model_type == 'CNN':
        return train_test_split(CNNXdata, CNNYdata, test_size=0.2, random_state=1)
    if model_type == 'ANN':
        return train_test_split(NNXdata, NNYdata, test_size=0.2, random_state=1)

def getBMData_split(path, model_type='CNN'):
    CNNXtest_file_name = path + "/test_x.npy"
    Ytest_file_name = path + "/test_y.npy"
    CNNXtrain_file_name = path + "/train_x.npy"
    Ytrain_file_name = path + "/train_y.npy"

    CNNXtest = np.load(CNNXtest_file_name, allow_pickle=True)  # 데이터 로드. @파일명
    CNNXtrain = np.load(CNNXtrain_file_name, allow_pickle=True)
    Ytest = np.load(Ytest_file_name)  # 데이터 로드. @파일명
    Ytrain = np.load(Ytrain_file_name)

    if CNNXtest.ndim == 3:
        CNNXtest = CNNXtest.reshape(CNNXtest.shape[0], CNNXtest.shape[1], CNNXtest.shape[2], 1)
    else:
        CNNXtest = CNNXtest

    if CNNXtrain.ndim == 3:
        CNNXtrain = CNNXtrain.reshape(CNNXtrain.shape[0], CNNXtrain.shape[1], CNNXtrain.shape[2], 1)
    else:
        CNNXtrain = CNNXtrain

    if Ytest.ndim == 3:
        Ytest = Ytest.reshape(Ytest.shape[0], (Ytest.shape[1] * Ytest.shape[2]))
    else:
        Ytest = Ytest
    if Ytrain.ndim == 3:
        Ytrain = Ytrain.reshape(Ytrain.shape[0], (Ytrain.shape[1] * Ytrain.shape[2]))
        Ytrain = Ytrain.reshape(Ytrain.shape[0])
    else:
        Ytrain = Ytrain

    CNNXtest = expand_3D(CNNXtest)
    CNNXtrain = expand_3D(CNNXtrain)
    Ytest = tf.keras.utils.to_categorical(Ytest)
    Ytrain = tf.keras.utils.to_categorical(Ytrain)

    if model_type == 'CNN':
        return CNNXtrain, CNNXtest, Ytrain, Ytest

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

    # plot graph of model
    plot_model(model, to_file='MNV2model.png', show_shapes=True)

    return model

##############################################################################
