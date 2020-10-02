#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import shutil
from Database_Class import Database_Raw

def Create_folder(path_to_folder, folder_name) :

    directory = os.path.join(path_to_folder, folder_name)
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

def Create_Raw_database_test(test_size, path_to_Raw_DB, Raw_DB_name) :
    '''
    Crée une toute petite base de données pour faire des tests rapides.
    les exemples sont choisis au hasard dans a grosse base de données
    :param test_size: taille de la base de test
    :param path_to_Raw_DB: bla
    :param Raw_DB_name: bla
    :param path_to_Raw_DB_test: bla
    :param Raw_DB_test_name: bla
    :return:
    '''
    Raw_DB = Database_Raw(Raw_DB_name, path_to_Raw_DB)

    path_to_Raw_DB_test = path_to_Raw_DB
    Raw_DB_test_name = ''.join([Raw_DB_name, '_{}'.format(test_size)])

    Create_folder(path_to_Raw_DB_test, Raw_DB_test_name)

    list_ID_test = np.random.randint(1, Raw_DB.size, test_size)

    for i in range(1, test_size+1) :

        # Source path
        src = ''.join([path_to_Raw_DB, Raw_DB_name, '/{}'.format(i)])
        # Destination path
        dest = ''.join([path_to_Raw_DB_test, Raw_DB_test_name, '/{}'.format(i)])

        # Copy the content of
        # source to destination
        destination = shutil.copytree(src, dest)

if __name__ == '__main__' :

    Create_Raw_database_test(test_size=10000,
                             path_to_Raw_DB='/home/abel/Desktop/VoxNet_Abel/data/',
                             Raw_DB_name='datasets')