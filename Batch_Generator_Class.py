from memory_profiler import profile

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import glob

# Creation of the BatchGenerator

class BatchGenerator(object):
    '''
    Cette classe génère un batch
    '''

    def __init__(self, path_to_data, list_IDs, batch_size):
        '''
        :param path_to_data: Emplacement des données d'entrée
        :param list_IDs: liste des idéntifiants des subdirectories contenant les exemples du Batch à génerer
        :param batch_size: taille du batch à générer
        :param dim: Dimension des images 3D en entrée (par défaut 30*30*30 et ça ne changera pas à priori..)
        :param n_classes: Nombre de classes du classifieur (par défaut 12 pour la classification par cond env.)
        '''
        'Initialization'
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.path_to_data = path_to_data

    def get_Batch_raw_paramater(self):
        '''
        :return: Génère le batch correspondant aux index donnés dans la list_IDs.
        Renvoie les paramètres eau et environnement brutes.
        '''

        X = []
        y = []

        for i, ID in enumerate(self.list_IDs):

            x = np.load(self.path_to_data + str(ID) + '/imgabswood.npy')
            X.append(x)

            y.append(np.load(self.path_to_data + str(ID) + '/targets_paramenv.npy'))

        X = np.asarray(X)
        X = tf.reshape(X, [-1, 30, 30, 30, 1])
        
        y = np.asarray(y)
        y = tf.reshape(y, [len(self.list_IDs), 2])
        return X, y
    
    def get_y_test(self):
        '''
        :return: Génère y_test.
        '''
        y = []
        
        for i, ID in enumerate(self.list_IDs):
            
            y.append(np.load(self.path_to_data + str(ID) + '/targets_paramenv.npy'))
        
        y = np.asarray(y)
        return y



