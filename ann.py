"""
Churn-rate prediction experiment
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

import prep_data as data
import hyper as hy


class ANN(object):

    def __init__(self,
                 num_layers=hy.num_layers,
                 kernel_initializer=hy.kernel_initializer,
                 activation=hy.activation,
                 batch_size=hy.batch_size,
                 nb_epoch=hy.nb_epoch,
                 optimizer=hy.optimzer,
                 use_pretrained=False,
                 optimize=False,
                 testcase=False,
                 testcase_data=None,
                 ):

        if optimize is True:
            best_params, best_accuracy = self.optimize_params(self)
            with open('savestate/optimal', 'w') as optimzed_file:
                optimzed_file.writelines(best_params)
                optimzed_file.writelines(best_accuracy)

        if use_pretrained is False:
            classifier = self.build_classifier(num_layers, kernel_initializer, activation, optimizer)
            classifier.fit(data.X_train, data.y_train, batch_size, nb_epoch)
            model_json = classifier.to_json()
            with open('savestate/model.json', 'w') as json_file:
                json_file.write(model_json)
            classifier.save_weights('savestate/model.h5')
        else:
            model_json = open('savestate/model.json', 'r')
            load_model = model_json.read()
            model_json.close()
            classifier = model_from_json(load_model)
            classifier.load_weights('savestate/model.h5', 'r')
            classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
            print("Modell erfolgreich geladen!\n")

    @staticmethod
    def build_classifier(num_layers, kernel_initializer, activation, optimizer):
        """
        TODO: Bias init ??
        """
        classifier = Sequential()
        for x in range(num_layers):
            if x == 0:
                classifier.add(Dense(6, kernel_initializer=kernel_initializer, activation=activation, input_dim=12))
                classifier.add(Dropout(0.1))
            else:
                classifier.add(Dense(6, kernel_initializer=kernel_initializer, activation=activation))

        classifier.add(Dense(1, kernel_initializer='glorot_normal', activation='sigmoid'))
        classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return classifier

    @staticmethod
    def build_opt_classifier(self, optimizer):
        """
        Builds the classifier for GridSearchCV
        :param self:
        :param num_layers:
        :param kernel_initializer:
        :param activation:
        :param optimizer:
        :return: classifier for Keras

        TODO: Bias init ??
        """
        classifier = Sequential()
        for x in range(hy.num_layers):
            if x == 0:
                classifier.add(Dense(6, kernel_initializer='glorot_uniform', activation='relu', input_dim=12))
                classifier.add(Dropout(0.1))
            else:
                classifier.add(Dense(6, kernel_initializer='glorot_uniform', activation='relu'))

        classifier.add(Dense(1, kernel_initializer='glorot_normal', activation='sigmoid'))
        classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return classifier

    @staticmethod
    def optimize_params(self):
        params_to_optimze = {
            'batch_size': [8, 16, 32],
            'nb_epoch': [100, 250],
            'optimizer': ['adam', 'rmsprop', 'adadelta'],
        }
        classifier = KerasClassifier(build_fn=self.build_opt_classifier)
        grid_search = GridSearchCV(estimator=classifier,
                                   param_grid=params_to_optimze,
                                   scoring='accuracy',
                                   cv=10)
        grid_search = grid_search.fit(X=data.X_train, y=data.y_train)
        return grid_search.best_params_, grid_search.best_score_


if __name__ == '__main__':
    ANN(use_pretrained=True)



"""
model_json = classifier.to_json()
with open('savestate/model.json', 'w') as json_file:
    json_file.write(model_json)
classifier.save_weights('savestate/model.h5')

y_pred = classifier.predict(X_test)

test_pred = classifier.predict(sc.fit_transform(np.array([[0, 0, 0, 0, 600, 1, 40, 60000, 2, 1, 1, 50000]])))
test_pred = (test_pred > 0.5)
print("mock pred: ", test_pred)


def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(6, kernel_initializer='glorot_uniform', activation='relu', input_dim=12))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(6, kernel_initializer='glorot_uniform', activation='relu'))
    classifier.add(Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


nclassifier = KerasClassifier(build_fn=build_classifier, batch_size=10, nb_epoch=100)
# K-Fold X Validation --Memo-- n_jobs -1 alle cpus
accuracies = cross_val_score(estimator=nclassifier, X=X_train, y=y_train, cv=10, n_jobs=-1, )
mean = accuracies.mean()
variance = accuracies.std()
print(mean, variance)


classifier = KerasClassifier(build_fn=build_classifier)
parameters = {
    'batch_size': [8, 16, 32],
    'nb_epoch': [100, 250],
    'optimizer': ['adam', 'rmsprop', 'adadelta'],
}

grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10)

grid_search = grid_search.fit(X=data.X_train, y=data.y_train)
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_

print(best_params, best_accuracy)

"""