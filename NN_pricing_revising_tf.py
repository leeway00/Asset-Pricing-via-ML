from pricing_data import krx_data
from MLutils.TensorUtils import *
from tensorflow_addons.metrics import r_square
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
import csv
np.set_printoptions(precision=4)
tf.keras.backend.clear_session()
tf.random.set_seed(77)


class CustomAccuracy(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        ind1 = tf.cast(y_true > 0, tf.float32)
        ind2 = tf.cast(y_pred > 0, tf.float32)
        return tf.reduce_mean(tf.square(y_true - y_pred) +
                              1/2 * tf.square(y_true*ind1 - y_pred*ind1) +
                              1/2 * tf.square(y_true*ind2 - y_pred*ind2))


class R2(tf.keras.metrics.Metric):
    def __init__(self):
        self.rsquared = self.add_weight(name='rs', initializer='zeros')
        self.dtype = tf.float32

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, self.dtype)
        y_pred = tf.cast(y_pred, self.dtype)
        self.rsquared.assign_add(1 - ((y_true-y_pred)**2).sum() / ((y_true - y_true.mean())**2).sum())

    def result(self):
        return self.true_positives


class NeuralNets:
    def __init__(self, year, y, x, y_test, x_test):
        self.y = y.values.reshape(-1, 1)
        self.x = x.values
        self.y_test = y_test.values.reshape(-1, 1)
        self.x_test = x_test.values
        self.year = year

        self.models = ['Net1', 'Net2', 'Net3', 'Net4', 'Net5', ]
        self.res_cols = ['year', 'name', 'params', 'pred_R2_OOS', 'train_score',
                         'CV_R2', 'CV_MSE', 'complexity']

        self.res = []
        self.error = []

    def compile_and_fit(self, model, name, patience=10, epochs=100, tensorboard=True):
        model.compile(loss='mse', optimizer='adam',
                      metrics=['mse', r_square.RSquare()])

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       patience=patience,
                                                       mode='min')

        log_dir = "logs/fit/name/" + datetime.now().strftime("%Y/%m/%d-%H:%M:%S")
        if tensorboard:
            tensorboard_callback = keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=1)
            callbacks = [early_stopping, tensorboard_callback]
        else:
            callbacks = [early_stopping]

        history = model.fit(self.x, self.y, epochs=epochs,
                            validation_data=(self.x_test, self.y_test),
                            callbacks=callbacks)
        return history

    def execute(self, model, params, name):

        pred_score = r_os(self.y_test, reg.predict(self.x_test))

        # year, name, best CV result, prediction score (R^2 OOS),
        #   training set score(R^2), CV result(R^2), CV result(MSE), complexity
        self.res.append([self.year, name, reg.best_params_, pred_score,
                        reg.score(self.x, self.y), reg.best_score_,
                        -reg.cv_results_['mean_test_neg_mean_squared_error'], complexity])
        return reg

    def Net1(self):
        net = keras.Sequential([
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='relu'),
        ])
        net = self.compile_and_fit(net, 'Net1')
        return net
    
    def Net2(self):
        net = keras.Sequential([
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='relu')
        ])
        net = self.compile_and_fit(net, 'Net2')
        return net

    def Net3(self):
        net3 = keras.Sequential([
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(1, activation='relu')
        ])
        net = self.compile_and_fit(net, 'Net3')
        return net

    def Net4(self):
        net1 = keras.Sequential([
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(4, activation='relu'),
            keras.layers.Dense(1, activation='relu')
        ])
        net = self.compile_and_fit(net, 'Net4')
        return net


    def fit_all(self):
        try:
            for i in self.models:
                getattr(self, i)
        except:
            print(f'Error while model {i} in {self.year}')
            self.error.append((self.year, i))

        with open('error.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerows(self.error)

        result = pd.DataFrame(self.res, columns=self.res_cols)
        if 'result.csv' in os.listdir():
            result.to_csv('result_NN.csv', mode='a', header=False, index=False)
        else:
            result.to_csv('result_NN.csv', index=False)

        self.result = result

    @property
    def pricing_result_(self):
        if 'result' in dir(self):
            return self.result


if __name__ == '__main__':
    train, test, test_years = krx_data()

    start = True
    pbar = tqdm(test_years)
    # for year in pbar:
    #     pbar.set_description(f"Processing {year}")
    #     y, x = train[year]
    #     y_test, x_test = test[year]
    #     if start == True:
    #         mod = Regressions(year, x, y, x_test, y_test)
    #         start = False
    #     else:
    #         mod.reset(year, x, y, x_test, y_test)

    #     mod.fit_all()
