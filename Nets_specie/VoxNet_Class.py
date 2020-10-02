# -*- coding: utf-8 -*-
from memory_profiler import profile

import gc

import numpy as np
import tensorflow as tf
import os

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Softmax
from tensorflow.keras.losses import categorical_crossentropy, sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from Nets.Batch_Generator_Class import BatchGenerator
from Database_Class import Database

class VoxNet(object) :

    def __init__(self, n_classes):
        '''
        :param n_classes: Nombre de classe en sortie
        '''
        self.n_classes = n_classes

        self.model = Sequential()
        self.model.add(Conv3D(30, 5, 2, activation='relu', input_shape=(30, 30, 30, 1)))
        self.model.add(Conv3D(30, 3, 1, activation='relu'))
        self.model.add(MaxPooling3D(pool_size=(2, 2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(self.n_classes, activation='softmax'))

    def Compile(self, loss_function = sparse_categorical_crossentropy, optimizer = Adam(), metrics=['accuracy']):
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

    def get_db(self, DB_name, path_to_DB) :
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

    def split_db(self, test_size) :
        '''
        Cette fonction découpe la base de données en 2, une partie pour l'apprentissage et une partie
        pour le test. La base de test aura la meme taille qu'un step d'apprentissage. Cela correspond
        à la taille de BD maximale que l'on peut garder en mémoire vive sur cet ordinateur.
        :test_size: Taille de la base de test. Il est conseillé de la choisir inférieure à
        200 exemples pour un ordinateur avec une RAM de 8Go
        :return: Définit deux listes d'index aléatoires de la base de données. Une pour le test
        et une pour l'entrainement.
        '''

        self.test_size = test_size
        self.train_size = self.db.size - self.test_size

        self.IDs_test = np.random.randint(0, self.db.size, test_size)
        self.IDs_train = []
        for k in range(self.db.size) :
            if k not in self.IDs_test :
                self.IDs_train.append(k)

    def fit_one_step(self):
        '''
        Charge le model, l'entraine sur un nombre défini de batchs, puis sauvegarde le modèle.
        :param list_IDs: liste de taille n_batch_per_step contenant les listes des index des
        exemples de chaque batch sur lequel fit_one_step entraine le réseau.
        :return:
        '''

        self.model = keras.models.load_model(self.path_model)
        #self.model.compile(loss=self.loss_function,
        #                   optimizer=self.optimizer,
        #                   metrics=self.metrics)

        for i in range(self.n_batch_per_step):

            self.Batch = BatchGenerator(self.path_to_data, self.list_IDs[i], batch_size=self.batch_size, n_classes=self.n_classes)
            self.X, self.y = self.Batch.get_Batch()

            self.model.fit(self.X, self.y, epochs=5, steps_per_epoch=12)
            #self.train_on_batch(self.X, self.y)

        self.model.save(self.path_model)
        self.iter += 1
        
        if self.iter%10 == 0 :
            self.model.save(''.join([self.path_to_weights, 'model_iter_{}.h5'.format(self.iter)]))

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
        while n_step < (self.train_size-self.step_size)/self.step_size:

            self.list_IDs = [[self.IDs_train[n_step*self.step_size + j*self.batch_size + i]
                         for i in range(self.batch_size)] for j in range(self.n_batch_per_step)]

            self.fit_one_step()
            n_step += 1

    def test_one_step(self, no_epoch) :
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
        self.model.compile(loss=sparse_categorical_crossentropy,
                      optimizer=Adam(),
                      metrics=['accuracy'])

        Batch = BatchGenerator(self.path_to_data, self.IDs_test, self.batch_size, self.n_classes)
        X_test, y_test = Batch.get_Batch_test()

        y_pred = self.model.predict(X_test, verbose=1, steps=1)
        y_pred = [list(y).index(max(y)) for y in y_pred]

        con_mat = tf.confusion_matrix(labels=y_test, predictions=y_pred, num_classes=self.n_classes)

        score = self.model.evaluate(X_test, y_test, verbose=0, steps=1)
        print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

        con_mat_name = 'ConMattob_iter_{}'.format(self.iter)
        con_mat = con_mat.eval(session=tf.Session())

        np.save(os.path.join(self.path_to_metrics, con_mat_name), con_mat)

    def sommaire(self) :
        self.model.summary()

    # Function to fit model on n_epochs epochs

    def train_and_monitor(self, path_to_data, path_model, path_to_metrics, path_to_weights, batch_size, n_batch_per_step, n_epochs, n_classes, validation_split):

        self.path_model = path_model
        self.path_to_data = path_to_data
        self.path_to_metrics = path_to_metrics
        self.path_to_weights = path_to_weights
        self.batch_size = batch_size
        self.n_batch_per_step = n_batch_per_step
        self.n_epochs = n_epochs
        self.no_epoch = 0
        self.n_classes = n_classes
        self.validation_split = validation_split
        self.step_size = self.n_batch_per_step * self.batch_size
        self.iter = 0

        self.Batch_test = BatchGenerator(self.path_to_data, self.IDs_test, batch_size=self.batch_size,
                                         n_classes=self.n_classes)
        self.X_test, self.y_test = self.Batch_test.get_Batch_test()

        self.model.save(self.path_model)


        for i in range(n_epochs):

            self.model = keras.models.load_model(self.path_model)
            self.fit_one_epoch()
            self.no_epoch += 1
            print('End epoch {}'.format(i))
            self.test_one_step(i)
            self.model.save(''.join([self.path_to_weights, 'model_epoch_{}.h5'.format(i)]))

            del self.model
            gc.collect()
            tf.keras.backend.clear_session()
            tf.reset_default_graph()
