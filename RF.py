####
import matplotlib.pylab as plt
import numpy as np

Source_dirSensor = "BMData"  ## 수정

CNNXdata_file_name = Source_dirSensor + "/xdata.npy"
NNXdata_file_name = Source_dirSensor + "/xdata2.npy"
Ydata_file_name = Source_dirSensor + "/ydata.npy"

CNNXdata = np.load(CNNXdata_file_name, allow_pickle=True)  # 데이터 로드. @파일명
NNXdata = np.load(NNXdata_file_name)  # 데이터 로드. @파일명
Ydata = np.load(Ydata_file_name)  # 데이터 로드. @파일명
Ydata.dtype
Ydata.shape

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

####

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(NNXdata, NNYdata, test_size=0.2, random_state=1)

from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=5, verbose=2,
                               random_state=42, n_jobs=2)
# Fit the random search model
rf_random.fit(X_train, y_train)

rf_random.best_params_


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy


best_random_rf = rf_random.best_estimator_
Fin_accuracy = evaluate(best_random_rf, X_test, y_test)