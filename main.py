import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.metrics import accuracy_score
from tensorflow import keras
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

learn_rate_opts = (0.00001, 0.0001)

def create_model_Bayesian(base_learning_rate, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test):

    model = utils.create_model(base_learning_rate=base_learning_rate)

    model.fit(x_train, y_train, epochs=initial_epochs, batch_size=batch_size, verbose=2)

    y_pred = model.predict(x_test)

    # evaluate accuracy for classification model
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    acc = accuracy_score(y_test, y_pred, normalize=False)
    loss = 1 - acc

    return loss

MNV2_param_options1 = {
    'base_learning_rate': learn_rate_opts,
}

model_MNV2pti = BayesianOptimization(
    f=create_model_Bayesian,
    pbounds=MNV2_param_options1,
    verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1234,
)

model_MNV2pti.maximize(
    init_points=0,
    n_iter=10, acq='ei', xi=0.01
)

for i, res in enumerate(model_MNV2pti.res):
    print('Iteration {}: \n\t{}'.format(i, res))
print("Best result: {}; f(x) = {}.".format(model_MNV2pti.max["params"], model_MNV2pti.max["target"]))

# Create model
model = utils.create_model(img_size=img_size,
                           momentum=momentum,
                           classes=classes,
                           base_learning_rate=model_MNV2pti.max["params"]['base_learning_rate'])

# Print model architecture
model.summary()

loss0, accuracy0 = model.evaluate(x_test, y_test, batch_size=batch_size, steps=validation_steps)
print("\nInitial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

# Train for epochs
history = model.fit(x_train, y_train,
                    epochs=initial_epochs,
                    batch_size=batch_size,
                    validation_data=(x_test, y_test),
                    verbose=2)

# todo save weight as .h5 file
# model_path = 'MobileNet.h5'
# Save model
# model.save(model_path)
# Reload model
# new_model = keras.models.load_model(model_path)
# new_model.summary()
# Check accuracy
# loss, acc = new_model.evaluate(x_test, y_test, verbose=2)
# print('복원된 모델의 정확도: {:5.2f}%'.format(100 * acc))

utils.show_result(history)
utils.show_confusion_matrix(model, x_test, y_test)
