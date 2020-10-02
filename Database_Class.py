#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on Thu Jul 23 10:23:09 2020

@author: abel

Description : Définition de la classe DataBase en entrée du réseau VoxNet Cette classe a pour vocation de créer
la base de données et de fournir toutes les méthodes utiles à sa complétion, à son nettoyage et à sa transformation.

TODO : Rajouter les visualisations dans les méthodes. + Analyse de la base si besoin est.
'''

# Import des bibliothèques utiles
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd

# Import des fonctions utiles
from Utils_specie.Conversion import convert_vct_to_npy, convert_csv_to_npy, create_label_env, create_label_random, isolate_paramenv

# Définition de la classe Database_Raw
class Database_Raw(object) :
    def __init__(self, DB_name, path_to_DB):
        '''
        :param DB_name: Nom de la base de données brutes
        :param path_to_DB: Emplacement de la base de données brutes
        '''
        self.DB_name = DB_name
        self.path_to_DB = path_to_DB
        path = os.path.join(self.path_to_DB, self.DB_name)
        self.path = path

        # On compte le nombre d'exemples dans la base de données brutes
        sub_path = ''.join([self.path, '/*'])
        self.size = len(glob.glob(sub_path))

        #On récupère les traitements précédents s'il y en a eu un
        ID_name = ''.join([self.DB_name, '_IDs_Elabores.npy'])
        if os.path.exists(os.path.join(self.path_to_DB, ID_name)) :
            IDs_Elabores = np.load(os.path.join(self.path_to_DB, ID_name))
            self.IDs_Elabores = IDs_Elabores
        else :
            self.IDs_Elabores = []

    def Content(self):
        '''
        :return: Cette fonction permet d'accéder à la liste des fichiers présents dans chaque subdirectory (ou exemple)
        de la base de données brutes.
        '''

        [x,y,z] = np.random.randint(0, int(self.size), 3)
        for k in [x,y,z] :
            sub_path = ''.join([self.path, '/{}'.format(k)])
            files = os.listdir(sub_path)
            print('Exemple {}'.format(k))
            print(files)

    def Tri(self, files_to_find):
        '''
        Pas très élégant et très couteux en temps comme façon de faire. Parce que les fichier targets.csv ne sont pas tous construits pareil.
        Pour gagner du temps on peut diviser le problème et ne prendre que les paramètres environnement puis que les paramètres strats.
        Attention il faudra alors modifier la fonction conversion_csv_to_npy dans utils.conversion.
        :param files_to_find: liste des fichiers (nom en str) qu'on s'attend à trouver dans chacun des subdirectories d'exemple.
        :return: Trie la base de données brutes, construit et sauvegarde une liste des index de subdirectories 'propres' c'est-à-dire contenant tout les fichiers
        nécessaires, sans qu'aucun ne soit cassé. Cette liste est sauvegardé au format .npy dans le dossier contenant la base de données brutes.
        '''

        IDs_Elabores = []

        for i in range(1, self.size + 1):

            test = True
            sub_path = ''.join([self.path, '/{}/'.format(i)])
            parameter_file = os.path.join(sub_path, 'targets.csv')

            CritTri1 = [os.path.exists(os.path.join(sub_path, file)) for file in files_to_find]
            if False in CritTri1 :
                test = False

            try :

                # Test le remplissage des fichiers paramètres
                df_parameters = pd.read_csv(parameter_file, header=None, sep=' ').T

                l = [float(df_parameters[0][1]), float(df_parameters[1][1]), float(df_parameters[2][1]), float(df_parameters[2][2]), float(df_parameters[2][3]), float(df_parameters[3][1]),float(df_parameters[3][2]), float(df_parameters[3][3]), float(df_parameters[4][1]), float(df_parameters[5][1]), float(df_parameters[5][2]), float(df_parameters[5][3]),float(df_parameters[6][1]), float(df_parameters[6][2]), float(df_parameters[6][3]), float(df_parameters[7][1])]

                #Test ultime
                if not(float(df_parameters[1][1]) / 200 in [0.5, 0.75, 1]) or not(float(df_parameters[0][1]) in [0.4, 0.6, 0.8, 1]):
                    test = False

            except :
                test = False
                pass

            if test == True :
                IDs_Elabores.append(i)

            if i%1000 == 0:
                print('Tri completed : {}'.format(100 - ((self.size - i)/self.size)*100))

        ID_name = ''.join([self.DB_name, '_IDs_Elabores.npy'])
        np.save(os.path.join(self.path_to_DB, ID_name), IDs_Elabores)
        self.IDs_Elabores = IDs_Elabores

    def Tri_aleatoire(self, files_to_find):

            IDs_Elabores = []

            for i in range(1, self.size + 1):

                test = True
                sub_path = ''.join([self.path, '/{}/'.format(i)])
                parameter_file = os.path.join(sub_path, 'targets.csv')

                CritTri1 = [os.path.exists(os.path.join(sub_path, file)) for file in files_to_find]
                print(CritTri1)
                print(parameter_file)
                if False in CritTri1:
                    test = False

                try:

                    # Test le remplissage des fichiers paramètres
                    df_parameters = pd.read_csv(parameter_file, header=None, sep=' ').T
                    print(df_parameters)

                    l = [float(df_parameters[0][1]), float(df_parameters[1][1])]

                    # Test ultime
                    if not (float(df_parameters[1][1])/200 > 0.5 and float(df_parameters[1][1])/200 < 1) or not (
                            float(df_parameters[0][1]) > 0.5 and float(df_parameters[0][1]) < 1) :

                        test = False
                        print('Condition contenu == float : '+str(test))

                except:
                    print('Erreur :' +str(i))
                    test = False

                if test == True:
                    IDs_Elabores.append(i)

                if i % 1000 == 0:
                    print('Tri completed : {}'.format(100 - ((self.size - i) / self.size) * 100))

            ID_name = ''.join([self.DB_name, '_IDs_Elabores.npy'])
            np.save(os.path.join(self.path_to_DB, ID_name), IDs_Elabores)
            self.IDs_Elabores = IDs_Elabores

# Definition de la classe Database
class Database(object) :
    '''

    '''
    def __init__(self, DB_name, path_to_DB) :
        '''
        :param DB_name: Nom de la base de données
        :param path_to_DB: Emplacement de la base de données
        '''
        self.DB_name = DB_name
        self.path_to_DB = path_to_DB
        path = os.path.join(self.path_to_DB, self.DB_name)
        self.path = path

        #On compte le nombre d'exemple dans la base
        sub_path = ''.join([self.path, '/*'])
        self.size = len(glob.glob(sub_path))

    def Content(self):
        '''
        :return: Cette fonction permet d'accéder à la liste des fichiers présents dans chaque subdirectory (ou exemple)
        de la base de données brutes.
        '''
        if self.size != 0:

            x = np.random.randint(int(self.size))

            sub_path = ''.join([self.path, '/{}'.format(x)])
            files = os.listdir(sub_path)
            print('Exemple {}'.format(x))
            print(files)

        else :
            print('Empty Database')

    def Create(self):
        '''
        :param path_to_raw_data: Emplacement des données brutes pour la création de la base de données
        :return: Si la base de données n'existe pas déjà, cette méthode crée un dossier nommé DB_name à
        l'emplacement path_to_DB.
        '''
        directory = os.path.join(self.path_to_DB, self.DB_name)
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
            else :
                print('Il y a déjà une base de données à cet emplacement')
        except OSError:
            print('Error: Creating directory. ' + directory)

    def Fill_and_Preprocess(self, raw_db_name, path_to_raw_db):
        '''
        :param raw_db_name: Nom de la base de données brutes
        :param path_to_raw_db: Emplacement de la base de données brutes
        :param files_to_find: les fichiers que l'on doit trouver dans la base de données brutes.
        Par défaut, ces fichiers seront les deux fichiers .vct et .csv. Attention, si on veut
        mettre autre chose, il faut aussi changer les fonctions de conversion utilisées.
        :return: Création d'un subdirectory par subdirectory de données brutes dont la qualité a été vérifiée. Les
        subdirectories ainsi créés auront pour noms les entiers consécutifs de 0 à à la taille de la base de donnéees.
        (Attention le consécutif est très important, la plupart des méthodes ci-après reposent sur cette nomenclature)
        '''

        # On crée une instance de la classe Database_Raw à l'aide de l'emplacement et du nom de la base
        # de données brutes en entrée.
        raw_db = Database_Raw(raw_db_name, path_to_raw_db)

        ID_name = ''.join([raw_db_name, '_IDs_Elabores.npy'])
        if os.path.exists(os.path.join(path_to_raw_db, ID_name)) :
            print('ok')
            IDs_Elabores = np.load(os.path.join(path_to_raw_db, ID_name))
            raw_db.IDs_Elabores = IDs_Elabores
            print(IDs_Elabores)
        else :
            print('non pas ok')
            print(os.path.join(path_to_raw_db, ID_name))
            raw_db.Tri(['imgabswood.vct', 'targets.csv'])
            IDs_Elabores = np.load(os.path.join(path_to_raw_db, ID_name))
            raw_db.IDs_Elabores = IDs_Elabores

        # On intitialise le nom des futurs subdirectories de la base à 0.
        ID = 0
        print('{} Exemples à construire'.format(len(raw_db.IDs_Elabores)))

        for index in raw_db.IDs_Elabores :

            subdirID = ''.join([self.path, '/{}'.format(ID)])
            try:
                if not os.path.exists(subdirID):
                    os.makedirs(subdirID)
            except OSError:
                print('Error: Creating directory. ' + directory)

            path_target_file = ''.join([raw_db.path, '/{}/'.format(index)])
            path_image_file = ''.join([raw_db.path, '/{}/'.format(index)])

            path_image_to_save = ''.join([self.path, '/{}/'.format(ID)])
            path_target_to_save = ''.join([self.path, '/{}/'.format(ID)])

            convert_vct_to_npy(path_image_file, path_image_to_save, vct_image_name='imgabswood.vct', npy_image_name='imgabswood.npy')
            convert_csv_to_npy(path_target_file, path_target_to_save, csv_target_name='targets.csv', npy_target_name='targets.npy')

            if ID%1000 == 0 :
                print('Fill_and_Preprocess completed : {}%'.format(100 - ((len(raw_db.IDs_Elabores)- ID)/len(raw_db.IDs_Elabores))*100))

            ID+=1

        sub_path = ''.join([self.path, '/*'])
        self.size = len(glob.glob(sub_path))

    def Labelize_env(self, dict_label_env):
        '''
        Fonction pour rajouter les labels environnements dans la base de données à partir des fichiers targets.csv
        Deux fichiers seront rajoutés, un .txt contenant le label textuel (Ex : 'HWHL') et un .npy contenant la classe
        numérique (entier representant la classe de l'exmple), avec lequel la correspondance est établie grâce
        au dictionnaire dict_label_env.
        :param dict_label_env: Dictionnaire pour la classification par conditions environnementales (voir fonction
        create_label_env)
        :return: Création de deux fichiers label.txt et label.npy par subdirectory (exemple) de la BD.
        '''

        for i in range(self.size):
            print(i)

            path_npy_target = ''.join([self.path, '/{}/'.format(i)])

            create_label_env(path_npy_target, npy_target_name='targets.npy', VoxNet_class_dictionary=dict_label_env,
                                  class_name='targets_class.npy')

            if i%1000 == 0:
                print('Labelize : {}%'.format(
                100 - ((self.size - i) / self.size) * 100))

    def Labelize_strat(self, dict_label_strat):
        '''
        A définir encore
        :param dict_label_strat: Dictionnaire pour la classification par conditions environnementales
        :return:
        '''
        pass

    def Labelize_random_climate(self):

        for i in range(self.size):

            path_npy_target = ''.join([self.path, '/{}/'.format(i)])

            create_label_random(path_npy_target, npy_target_name='targets.npy',
                             class_name='targets_class.npy')

            if i % 1000 == 0:
                print('Labelize : {}%'.format(
                    100 - ((self.size - i) / self.size) * 100))

    def get_paramenv(self):

        for i in range(self.size) :
            path_npy = ''.join([self.path, '/{}/'.format(i)])
            isolate_paramenv(path_npy, npy_target_name='targets.npy', paramenv_name='targets_paramenv.npy')

            if i % 1000 == 0:
                print('Isolate : {}%'.format(
                    100 - ((self.size - i) / self.size) * 100))

if __name__ == '__main__' :

    DB = Database(DB_name='DB_test',
                  path_to_DB='/home/abel/Desktop/VoxNet_Specie_Sauvegarde/Data_specie/')
    DB.get_paramenv()