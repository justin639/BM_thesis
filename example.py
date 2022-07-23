# 평균제곱오차와 교차 엔트로피를 비교하는 프로그램

import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from tensorflow.keras import regularizers

import matplotlib.pyplot as plt

# MNIST 읽어 와서 신경망에 입력할 형태로 변환
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1)  # 텐서 모양 변환
x_test = x_test.reshape(10000, 28, 28, 1)
x_train = x_train.astype(np.float32) / 255.0  # ndarray로 변환
x_test = x_test.astype(np.float32) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)  # 원핫코드로 변환
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 하이퍼 매개변수 설정
batch_size = 128    # 연산 한번 할때 들어가는 데이터 크기
n_epoch = 10        # 순전파 역전파를 통해 한번 통과하는 횟수(학습 횟수)
k = 5  # 5-겹

default_loss_function = 'categorical_crossentropy'


# 신경망 구조 설정
# C-C-P-FC-FC 구조 컨볼루션 신경망 설계
# 모델을 설계해주는 함수(모델을 나타내는 객체 model을 반환)
def build_model(l2_reg, dropout_rate):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate[0]))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate[1]))
    model.add(Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(l2_reg)))
    return model


# 교차 검증을 해주는 함수(loss function 맞게)
def cross_validation(data_gen, dropout_rate, l2_reg):
    accuracy = []

    for train_index, val_index in KFold(k).split(x_train):
        xtrain, xval = x_train[train_index], x_train[val_index]
        ytrain, yval = y_train[train_index], y_train[val_index]
        dmlp = build_model(l2_reg, dropout_rate)
        dmlp.compile(loss=default_loss_function, optimizer=Adam(), metrics=['accuracy'])
        if data_gen:
            generator = ImageDataGenerator(rotation_range=3.0, width_shift_range=0.1, height_shift_range=0.1,
                                           horizontal_flip=True)
            dmlp.fit_generator(generator.flow(x_train, y_train, batch_size=batch_size), epochs=n_epoch,
                               validation_data=(x_test, y_test), verbose=2)
        else:
            dmlp.fit(xtrain, ytrain, batch_size=batch_size, epochs=n_epoch, validation_data=(x_test, y_test),
                     verbose=2)
        accuracy.append(dmlp.evaluate(xval, yval, verbose=0)[1] * 100)

    return accuracy


def validation(loss_function, visualize):
    dmlp = build_model(0.0, [0.0, 0.0])
    dmlp.compile(loss=loss_function, optimizer=Adam(), metrics=['accuracy'])
    hist = dmlp.fit(x_train, y_train, batch_size=128, epochs=30, validation_data=(x_test, y_test), verbose=2)

    # # 커널의 시각화
    if visualize:
        dmlp.summary()  # cnn 모델의 정보 출력

        for layer in dmlp.layers:
            if 'conv' in layer.name:
                kernel, biases = layer.get_weights()
                print(layer.name, kernel.shape)  # 커널의 텐서 모양을 출력

        kernel, biases = dmlp.layers[0].get_weights()  # 층 0의 커널 정보를 저장
        minv, maxv = kernel.min(), kernel.max()
        kernel = (kernel - minv) / (maxv - minv)
        n_kernel = 32

        plt.figure(figsize=(20, 3))
        plt.suptitle("Kernels of con2d_4")
        for i in range(n_kernel):
            f = kernel[:, :, :, i]
            plt.subplot(3, n_kernel, i+1)
            plt.imshow(f[:, :, 0], cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title(str(i))
        plt.show()

    res = dmlp.evaluate(x_test, y_test, verbose=0)[1] * 100
    return res, hist


# MSE와 CE를 실행
# res_mse, hist_mse = validation('mean_squared_error', False)
res_ce, hist_ce = validation('categorical_crossentropy', True)

# 데이터 증대, 드롭아웃 ,가중치 감쇠를 제거 조사로 실행
acc_000 = cross_validation(False, [0.0, 0.0], 0.0)
acc_001 = cross_validation(False, [0.0, 0.0], 0.01)
acc_010 = cross_validation(False, [0.5, 0.5], 0.0)
acc_011 = cross_validation(False, [0.5, 0.5], 0.01)
acc_100 = cross_validation(True, [0.0, 0.0], 0.0)
acc_101 = cross_validation(True, [0.0, 0.0], 0.01)
acc_110 = cross_validation(True, [0.5, 0.5], 0.0)
acc_111 = cross_validation(True, [0.5, 0.5], 0.01)

# MSE와 CE의 정확률 비교
print("MSE:", res_mse, "%")
print("CE:", res_ce, "%")

# 데이터 증대, 드롭아웃 가중치 감쇠의 성능 비교
print("출력 형식: [Data augmentation-Dropout-l2 regularizer] (교차검증 시도/평균)")
print("[000] (", acc_000, "/", np.array(acc_000).mean(), ")")
print("[001] (", acc_001, "/", np.array(acc_001).mean(), ")")
print("[010] (", acc_010, "/", np.array(acc_010).mean(), ")")
print("[011] (", acc_011, "/", np.array(acc_011).mean(), ")")
print("[100] (", acc_100, "/", np.array(acc_100).mean(), ")")
print("[101] (", acc_101, "/", np.array(acc_101).mean(), ")")
print("[110] (", acc_110, "/", np.array(acc_110).mean(), ")")
print("[111] (", acc_111, "/", np.array(acc_111).mean(), ")")

# 정확률 그래프
plt.plot(hist_mse.history['accuracy'])
plt.plot(hist_mse.history['val_accuracy'])
plt.plot(hist_ce.history['accuracy'])
plt.plot(hist_ce.history['val_accuracy'])
plt.title('Model accuracy comparison between MSE and cross entropy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train_mse', 'Validation_mse', 'Train_ce', 'Validation_ce'], loc='best')
plt.grid()
plt.show()

# 손실 함수 그래프
plt.plot(hist_mse.history['loss'])
plt.plot(hist_mse.history['val_loss'])
plt.plot(hist_ce.history['loss'])
plt.plot(hist_ce.history['val_loss'])
plt.title('Model loss comparison between MSE and cross entropy')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train_mse', 'Validation_mse', 'Train_ce', 'Validation_ce'], loc='best')
plt.grid()
plt.show()

# 세 규제기범의 정확률을 박스플롯으로 비교
plt.grid()
plt.boxplot([acc_000, acc_001, acc_010, acc_011, acc_100, acc_101, acc_110, acc_111],
            labels=["000", "001", "010", "011", "100", "101", "110", "111"])
plt.show()
