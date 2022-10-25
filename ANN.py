import pandas as pd
from bayes_opt import BayesianOptimization, UtilityFunction
import seaborn as sns
import warnings
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.layers import LeakyReLU, Input, Dense, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.losses import mean_absolute_error
from tensorflow.keras.optimizers import Nadam, Adam
from tensorflow.keras.utils import plot_model

import utils

warnings.filterwarnings("ignore")
path = "BMData"
hiddennode = 128
nb_layers = 1
dropout_rate = 0.2

l2_penalty = 0.0001
learning_rate = 0.001
hiddennode_rate = 2

n_class = 7

n_input = 12
Alpha = 0.01
Kernel_init = 'glorot_normal'

x_train, x_test, y_train, y_test = utils.getBMData(path + "/", "ANN")



def createNNModel2(hiddennode, dropout_rate,
                   l2_penalty, learning_rate, hiddennode_rate):
    leaky_relu = LeakyReLU(alpha=Alpha)

    visible = Input(shape=(n_input,))
    hidden1 = Dense(hiddennode, activation=leaky_relu, kernel_initializer=Kernel_init,
                    kernel_regularizer=regularizers.l2(l2_penalty))(visible)
    Drop1 = Dropout(dropout_rate)(hidden1)
    hidden2 = Dense(round(hiddennode / hiddennode_rate), activation=leaky_relu, kernel_initializer=Kernel_init,
                    kernel_regularizer=regularizers.l2(l2_penalty))(Drop1)
    Drop2 = Dropout(dropout_rate)(hidden2)
    hidden3 = Dense(round(hiddennode / hiddennode_rate), activation=leaky_relu, kernel_initializer=Kernel_init,
                    kernel_regularizer=regularizers.l2(l2_penalty))(Drop2)
    Drop3 = Dropout(dropout_rate)(hidden3)
    hidden4 = Dense(round(hiddennode / hiddennode_rate), activation=leaky_relu, kernel_initializer=Kernel_init,
                    kernel_regularizer=regularizers.l2(l2_penalty))(Drop3)
    Drop4 = Dropout(dropout_rate)(hidden4)

    # classification output
    out_clas = Dense(n_class, activation='softmax', name='out_clas')(Drop4)

    model = Model(inputs=visible, outputs=out_clas)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(lr=learning_rate, beta_1=0.99, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True),
                  metrics=['accuracy'])
    model.summary()

    # plot graph of model
    plot_model(model, to_file='model.png', show_shapes=True)

    # fit the keras model on the dataset
    # regression, classification
    model.fit(x_train, y_train, epochs=125, batch_size=32, verbose=2)

    # make predictions on test set
    yPredA = model.predict(x_test)


    # evaluate accuracy for classification model
    yPredA = np.argmax(yPredA, axis=-1).astype('int')
    acc = accuracy_score(y_test, yPredA, normalize=False)
    loss = 1 - acc

    return loss


hiddennode = (500, 1000)
hiddennode_rate_opts = (1.0, 4.0)
learn_rate_opts = (0.00001, 0.0001)
dropout_rate_opts = (0, 0.5)
l2_penalty_opts = (0.00001, 0.1)

NN_param_options2 = {
    'hiddennode': hiddennode,
    'learning_rate': learn_rate_opts,
    'hiddennode_rate': hiddennode_rate_opts,
    'dropout_rate': dropout_rate_opts,
    'l2_penalty': l2_penalty_opts,
}

model_NNOpti = BayesianOptimization(
    f=createNNModel2,
    pbounds=NN_param_options2,
    verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1234,
)

model_NNOpti.maximize(
    init_points=0,
    n_iter=10, acq='ei', xi=0.01
)

for i, res in enumerate(model_NNOpti.res):
    print('Iteration {}: \n\t{}'.format(i, res))
print("Best result: {}; f(x) = {}.".format(model_NNOpti.max["params"], model_NNOpti.max["target"]))


def createNNModelrefit(hiddennode, dropout_rate,
                       l2_penalty, learning_rate, hiddennode_rate):
    leaky_relu = LeakyReLU(alpha=Alpha)

    visible = Input(shape=(n_input,))
    hidden1 = Dense(hiddennode, activation=leaky_relu, kernel_initializer=Kernel_init,
                    kernel_regularizer=regularizers.l2(l2_penalty))(visible)
    Drop1 = Dropout(dropout_rate)(hidden1)
    hidden2 = Dense(round(hiddennode / hiddennode_rate), activation=leaky_relu, kernel_initializer=Kernel_init,
                    kernel_regularizer=regularizers.l2(l2_penalty))(Drop1)
    Drop2 = Dropout(dropout_rate)(hidden2)
    hidden3 = Dense(round(hiddennode / hiddennode_rate), activation=leaky_relu, kernel_initializer=Kernel_init,
                    kernel_regularizer=regularizers.l2(l2_penalty))(Drop2)
    Drop3 = Dropout(dropout_rate)(hidden3)
    hidden4 = Dense(round(hiddennode / hiddennode_rate), activation=leaky_relu, kernel_initializer=Kernel_init,
                    kernel_regularizer=regularizers.l2(l2_penalty))(Drop3)
    Drop4 = Dropout(dropout_rate)(hidden4)

    # classification output
    out_clas = Dense(n_class, activation='softmax', name='out_clas')(Drop4)

    model = Model(inputs=visible, outputs=out_clas)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Nadam(lr=learning_rate),
                  metrics=['accuracy'])
    model.summary()

    # plot graph of model
    plot_model(model, to_file='model.png', show_shapes=True)

    # fit the keras model on the dataset
    model.fit(x_train, y_train, epochs=250, batch_size=100, verbose=2)

    # make predictions on test set
    yPredA = model.predict(x_test)


    # evaluate accuracy for classification model
    yPredA = np.argmax(yPredA, axis=-1).astype('int')
    acc = accuracy_score(y_test[:, 1], yPredA, normalize=False)

    return model, yPredA


model, yPredABOOpti = createNNModelrefit(model_NNOpti.max["params"]['hiddennode'],
                                                       model_NNOpti.max["params"]['dropout_rate'],
                                                       model_NNOpti.max["params"]['l2_penalty'],
                                                       model_NNOpti.max["params"]['learning_rate'],
                                                       model_NNOpti.max["params"]['hiddennode_rate'])


# evaluate accuracy for classification model
# yPredABOOpti = argmax(yPredABOOpti, axis=-1).astype('int')
accBOOpti = accuracy_score(y_test[:, 1], yPredABOOpti)
print('Accuracy: %.3f' % accBOOpti)

NN_cnf_matrixBOOpti = confusion_matrix(y_test[:, 1], yPredABOOpti)

# Create heatmap from the confusion matrix
class_names = [0, 1]  # name  of classes
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(pd.DataFrame(NN_cnf_matrixBOOpti), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('NN Confusion matrix BO Opti')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
tick_marks = [0.5, 1.5]
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)