# -*- coding: utf-8 -*-import gc

import numpy as np
import tensorflow as tf
import os
import shutil
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Softmax
from tensorflow.keras.losses import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import mean_squared_error as mse

from Batch_Generator_Class import BatchGenerator
from Database_Class import Database

class Test(object) :
    def __init__(self, path_model, loss_function, optimizer, metrics, path_to_data, IDs_test, path_to_results):

        self.path_model = path_model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.metrics = metrics
        self.path_to_data = path_to_data
        self.IDs_test = IDs_test
        self.path_to_results = path_to_results

        self.batch_size = len(self.IDs_test)

    def predict(self):

        self.model = keras.models.load_model(self.path_model)
        self.model.compile(loss=self.loss_function,
                           optimizer=self.optimizer,
                           metrics=self.metrics)

        Batch = BatchGenerator(self.path_to_data, self.IDs_test, self.batch_size)
        X_test, y_test = Batch.get_Batch_raw_paramater()
        y_test_arr = Batch.get_y_test()

        y_pred = self.model.predict(X_test, verbose=1, steps=1)

        self.y_pred = y_pred
        self.y_test = y_test_arr

        score = self.model.evaluate(X_test, y_test, verbose=0, steps=1)

        self.loss = score[0]
        self.metrics = score[1]

    def split_water_light(self):

        self.y_test_water = [self.y_test[i][0] for i in range(len(self.y_test))]
        self.y_test_light = [self.y_test[i][1] for i in range(len(self.y_test))]

        self.y_pred_water = [self.y_pred[i][0] for i in range(len(self.y_pred))]
        self.y_pred_light = [self.y_pred[i][1] for i in range(len(self.y_pred))]

    def process(self):

        y_pred_water = list(np.copy(self.y_pred_water))
        y_test_water = list(np.copy(self.y_test_water))

        y_pred_light = list(np.copy(self.y_pred_light))
        y_test_light = list(np.copy(self.y_test_light))

        y_test_new_water = []; y_pred_new_water = []; y_test_new_light = []; y_pred_new_light = []

        for i in range(len(y_test_water)):
            index_water = y_test_water.index(min(y_test_water))
            index_light = y_test_light.index(min(y_test_light))

            y_test_new_water.append(y_test_water[index_water])
            y_pred_new_water.append(y_pred_water[index_water])
            y_test_new_light.append(y_test_light[index_light])
            y_pred_new_light.append(y_pred_light[index_light])

            y_test_water.remove(y_test_water[index_water])
            y_pred_water.remove(y_pred_water[index_water])
            y_test_light.remove(y_test_light[index_light])
            y_pred_light.remove(y_pred_light[index_light])
            # print(self.dict[y_pred_name])

        return y_test_new_water, y_pred_new_water, y_test_new_light, y_pred_new_light

    def trace_pred(self):

        fig, axes = plt.subplots(nrows=1, ncols=2)

        y_test_water, y_pred_water, y_test_light, y_pred_light = self.process()

        RMSE_water = mse(y_test_water, y_pred_water)
        RMSE_water = round(RMSE_water, 5)
        RMSE_light = mse(y_test_light, y_pred_light)
        RMSE_light = round(RMSE_light, 5)

        axes[0].plot(y_test_water, label='y_test')
        axes[0].plot(y_pred_water, label='y_pred')
        axes[0].set_title('prediction water', fontstyle='italic')
        axes[0].text(60, 0.5, 'RMSE = ' + str(RMSE_water), fontsize=8)

        axes[1].plot(y_test_light, label='y_test')
        axes[1].plot(y_pred_light, label='y_pred')
        axes[1].set_title('prediction light', fontstyle='italic')
        axes[1].text(60, 0.5, 'RMSE = ' + str(RMSE_light), fontsize=8)

        axes[1].legend()

        plt.suptitle('Prediction test', fontsize='large', fontweight='bold')
        fig_name = 'Prediction_test.png'
        plt.savefig(os.path.join(self.path_to_results, fig_name))

if __name__ == '__main__' :

    test = Test(path_model = '/home/abel/Desktop/VoxNet_Specie_Sauvegarde/Weights_specie/model_devine2_epoch_17.h5',
                loss_function = mean_squared_error,
                optimizer = Adam(),
                metrics = ['mean_absolute_percentage_error'],
                path_to_data = '/home/abel/Desktop/VoxNet_Specie_Sauvegarde/Data_specie/DB_test/',
                IDs_test = [i for i in np.random.randint(0, 958, 20)],
                path_to_results = '/home/abel/Desktop/VoxNet_Specie_Sauvegarde/Metrics_specie/')

    test.predict()
    test.split_water_light()
    test.trace_pred()
