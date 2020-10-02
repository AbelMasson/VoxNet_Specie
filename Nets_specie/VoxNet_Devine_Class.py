# -*- coding: utf-8 -*-
from memory_profiler import profile

import gc

import numpy as np
import tensorflow as tf
import os
import shutil

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Softmax
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam

from Batch_Generator_Class import BatchGenerator
from Database_Class import Database


class VoxNet(object):

    def __init__(self):
        '''
       	Definition du modèle.
        '''
        self.model = Sequential()
        self.model.add(Conv3D(30, 5, 2, activation='relu', input_shape=(30, 30, 30, 1)))
        self.model.add(Conv3D(30, 3, 1, activation='relu'))
        self.model.add(MaxPooling3D(pool_size=(2, 2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(2))

    def Compile(self, loss_function=mean_squared_error, optimizer=Adam(), metrics=['mean_absolute_percentage_error']):
        '''
        Compile le réseau
        :param loss_function: fontion loss, par défaut categorical_crossentropy mais peut etre redéfinie
        :param optimizer: optimisateur.. Chercher ce que c'est. Par défaut Adam() comme chez voxnet_master
        :param metrics: metrics.. Chercher ce que c'est. min_average_precision c'est possible ici ?
        :return:
        '''

        self.loss_function = loss_function
        self.optimizer = optimizer
        self.metrics = metrics

        self.model.compile(loss=self.loss_function,
                           optimizer=self.optimizer,
                           metrics=self.metrics)

    def get_db(self, DB_name, path_to_DB):
        '''
        Cette fonction permet de définir la base de données sur laquelle l'entrainement aura lieu.
        :param DB_name: Nom de la base de données
        :param path_to_DB: Emplacement de la base de données
        :return:
        '''
        self.db = Database(DB_name, path_to_DB)

        '''
        #sub_path = ''.join([self.db.path, '/*'])
        #self.db.size = len(glob.glob(sub_path))
        #print(self.db.size)
        '''

    def split_db(self, val_size):
        '''
        Cette fonction découpe la base de données en 2, une partie pour l'apprentissage et une partie
        pour la validation. La base de validation aura la meme taille qu'un step d'apprentissage. Cela correspond
        à la taille de BD maximale que l'on peut garder en mémoire vive sur cet ordinateur.
        :val_size: Taille de la base de validation. Il est conseillé de la choisir inférieure à
        200 exemples pour un ordinateur avec une RAM de 8Go
        :return: Définit deux listes d'index aléatoires de la base de données. Une pour la validation
        et une pour l'entrainement.
        '''

        self.val_size = val_size
        self.train_size = self.db.size - self.val_size

        self.IDs_test = np.random.randint(0, self.db.size, val_size)
        self.IDs_train = []
        for k in range(self.db.size):
            if k not in self.IDs_test:
                self.IDs_train.append(k)

    def fit_one_step(self):
        '''
        Charge le model, l'entraine sur un nombre défini de batchs, puis sauvegarde le modèle.
        :param list_IDs: liste de taille n_batch_per_step contenant les listes des index des
        exemples de chaque batch sur lequel fit_one_step entraine le réseau.
        :return:
        '''

        self.model = keras.models.load_model(self.path_model)
        # self.model.compile(loss=self.loss_function,
        #                   optimizer=self.optimizer,
        #                   metrics=self.metrics)

        for i in range(self.n_batch_per_step):
            self.Batch = BatchGenerator(self.path_to_data, self.list_IDs[i], batch_size=self.batch_size)
            self.X, self.y = self.Batch.get_Batch_raw_paramater()

            history = self.model.fit(self.X, self.y, epochs=5, steps_per_epoch=12)
            #self.train_on_batch(self.X, self.y)
            self.train_loss.append(history.history['loss'])
            self.train_metrics.append(history.history['mean_absolute_percentage_error'])

        self.model.save(self.path_model)
        self.iter += 1

        if self.iter % 10 == 0:
            self.model.save(''.join([self.path_to_weights, 'model_devine2_iter_{}.h5'.format(self.iter)]))

        del self.model
        gc.collect()
        tf.keras.backend.clear_session()
        tf.reset_default_graph()

    # Function to fit model on one epoch
    def fit_one_epoch(self):
        '''
        Effectue l'entrainement sur une epoch entière, ie fit_one_step répétée sur le nombre de steps nécessaires
        :return:
        '''

        self.list_IDs = []
        n_step = 0
        while n_step < (self.train_size - self.step_size) / self.step_size:
            self.list_IDs = [[self.IDs_train[n_step * self.step_size + j * self.batch_size + i]
                              for i in range(self.batch_size)] for j in range(self.n_batch_per_step)]

            self.fit_one_step()
            n_step += 1

    def test_one_step(self, no_epoch):
        '''
        Effectue un test sur le jeu de données test, et renvoie les metrics. Ici on a choisi la matrice de confusion
        mais d'autres metrics peuvent etre utilisées (Precision, Recall, F1-score etc..)
        :param no_epoch: numéro de l'epoch où on se trouve dans l'apprentissage.
        :param list_IDs: liste des index de self.IDs_test correspondant aux indexes des exemples de la base
        de test qui seront testés ici.
        :return: Calcule et sauvegarde pour une epoch données les metrics dans le dossier metrics.

        Attention on ne del pas le modèle ici puisqu'on le fait juste après à la fin de l'epoch.
        Si on veut faire plus d'un test par epoch, il faudra trouver le charger puis le del directement ici.
        '''

        self.model = keras.models.load_model(self.path_model)
        self.model.compile(loss=self.loss_function,
                           optimizer=self.optimizer,
                           metrics=self.metrics)

        Batch = BatchGenerator(self.path_to_data, self.IDs_test, self.batch_size)
        X_test, y_test = Batch.get_Batch_raw_paramater()

        y_pred = self.model.predict(X_test, verbose=1, steps=1)
        
        np.save(os.path.join(self.path_to_metrics, 'y_val_pred2_epoch_{}.npy'.format(no_epoch)), y_pred)
        
        score = self.model.evaluate(X_test, y_test, verbose=0, steps=1)
        self.val_loss.append(score[0])
        print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
        if no_epoch >= 1 and score[0] > self.val_loss[-2] :
            self.stop = True
        
        source = self.path_model
        destination = ''.join([self.path_to_weights, 'model_devine2_epoch_{}.h5'.format(no_epoch)])
        shutil.copyfile(source, destination)

    def sommaire(self):
        self.model.summary()

    # Function to fit model on n_epochs epochs

    def train_and_monitor(self, path_to_data, path_model, path_to_metrics, path_to_weights, batch_size,
                          n_batch_per_step, n_epochs):

        self.path_model = path_model
        self.path_to_data = path_to_data
        self.path_to_metrics = path_to_metrics
        self.path_to_weights = path_to_weights
        self.batch_size = batch_size
        self.n_batch_per_step = n_batch_per_step
        self.n_epochs = n_epochs
        self.no_epoch = 0
        self.step_size = self.n_batch_per_step * self.batch_size
        self.iter = 0
        self.val_loss = []
        self.train_loss = []
        self.train_metrics = []
        self.stop = False

        self.Batch_test = BatchGenerator(self.path_to_data, self.IDs_test, batch_size=self.batch_size)
        self.X_test, self.y_test = self.Batch_test.get_Batch_raw_paramater()
        y_test_arr = self.Batch_test.get_y_test()
        np.save(os.path.join(self.path_to_metrics, 'y_val2.npy'), y_test_arr)

        self.model.save(self.path_model)

        for i in range(n_epochs):
            self.model = keras.models.load_model(self.path_model)
            self.fit_one_epoch()
            self.no_epoch += 1
            print('End epoch {}'.format(i))
            
            self.test_one_step(i)
            if self.stop == True :
                print("Argh ! La loss ne diminue plus !")
            
            del self.model
            gc.collect()
            tf.keras.backend.clear_session()
            tf.reset_default_graph()

        np.save(os.path.join(self.path_to_metrics, 'val_loss_per_epoch2.npy'), self.val_loss)
        np.save(os.path.join(self.path_to_metrics, 'train_loss2.npy'), self.train_loss)
        np.save(os.path.join(self.path_to_metrics, 'train_metrics2.npy'), self.train_metrics)
