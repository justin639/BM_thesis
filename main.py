from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
import utils

# Todo extract the model build and calculate hyper-parameters
data_path = "NearFallPaper_DataCode"
path = "BMData"
model_tf = 'tensorflow'
batch_size = 16
img_size = 35
momentum = 0.9
# epoch마다 점점 줄여보기
base_learning_rate = 0.00001
validation_steps = 20
initial_epochs = 100
classes = 7
# Get Data from mat file
# test_x, test_y, train_x, train_y = utils.getdata_from_mat(data_path, img_size)
x_train, x_test, y_train, y_test = utils.getBMData(path + "/")

print(x_train.shape)
print(y_train.shape)
# print(x_train)
# Create model
model = utils.create_model(img_size=img_size,
                           momentum=momentum,
                           classes=classes,
                           base_learning_rate=base_learning_rate,
                           model_type=model_tf)

# Print model architecture
model.summary()

loss0, accuracy0 = model.evaluate(x_test, y_test, batch_size=batch_size, steps=validation_steps)
print("\nInitial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

# Train for epochs
history = model.fit(x_train, y_train,
                    epochs=initial_epochs,
                    validation_data=(x_test, y_test),
                    verbose=2)
# todo save weight as .h5 file
model_path = 'MobileNet.h5'
# Save model
model.save(model_path)
# Reload model
new_model = keras.models.load_model(model_path)
new_model.summary()
# Check accuracy
loss, acc = new_model.evaluate(x_test, y_test, verbose=2)
print('복원된 모델의 정확도: {:5.2f}%'.format(100 * acc))

utils.show_result(history)
utils.show_confusion_matrix(model, x_test, y_test)
