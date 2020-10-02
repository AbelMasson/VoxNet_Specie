#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on Thu Jul 23 10:23:09 2020
@author: abel

Description : Procédure à suivre à partir de la base de données brutes pour obtenir la base d'entrée du réseau.
'''

import os
import numpy as np
import glob
from Database_Class import Database, Database_Raw

'''
PARAMETRES A MODIFIER EN FONCTION DE LA BASE DE DONNEES A CONSTRUIRE
ATTENTION : Si l'on souhaite constituer une base de test, il faut découper 
en amont la base de données brutes en base de test et base d'entrainement 
puis lancez ce programme sur chacune des deux.

Pour lancer ce programme, ouvrez un terminal de commande, 
placez vous dans le dossier contenant ce script ( cd Path/To/Script )
puis tapez la commande : python ./Get_database.py
'''
#-----------------------------------------------------------------------------------------------------------------------

#Nom de la base de données brutes
Raw_DB_name = 'databaseclimat'
#Emplacement de la base de données brutes
path_to_Raw_DB = '/media/abel/databasetes/'
#Nom de la base de données d'entrée à construire A CHOISIR
DB_name = 'DB_specie_v1'
#Emplacement de la base de données d'entrée A CHOSIR
path_to_DB = '/home/abel/Desktop/VoxNet_Abel_Propre/data/'

#-----------------------------------------------------------------------------------------------------------------------

'''
A PARTIR D'ICI NE RIEN TOUCHER
'''

'''
Instanciation des deux classes pour nos bases de données,
Creation physique de la base de données en entrée du réseau
'''
#Creation d'une instance de la classe Database pour notre base à construire
#Creation d'une instance de la classe Database_Raw pour notre base de données brutes
DB_v1 = Database(DB_name, path_to_DB)
DB_Raw = Database_Raw(Raw_DB_name, path_to_Raw_DB)

#Creation physique de la base de données
sub_path = ''.join([DB_v1.path_to_DB, DB_v1_name])
if not os.path.exists(sub_path):
    DB_v1.Create()
else :
    print('La base de données est déjà créée')

'''
Test du contenu des deux bases
'''

print('Contenu de la base de données créée : ')
DB_v1.Content()
print('Contenu de la base de données brutes : ')
DB_Raw.Content()

'''
Tri sur la qualité des données de DB_raw.
'''

ID_name = ''.join([DB_Raw.DB_name, '_IDs_Elabores.npy'])
sub_path = ''.join([DB_Raw.path_to_DB, ID_name])
if not os.path.exists(sub_path):
    DB_Raw.Tri_aleatoire(['imgabswood.vct', 'targets.csv'])
else :
    print('La base de données brutes est déjà triée')

'''
Traitement des données brutes et remplissage de la base de donnnées d'entrée
'''

if DB_v1.size == 0 :
    DB_v1.Fill_and_Preprocess(Raw_DB_name, path_to_Raw_DB)
else :
    print("La base de données d'entrée est déjà remplie")

'''
Vérification du traitement et du remplissage
'''
print('Contenu de la base de données remplie : ')
DB_v1.Content()

'''
Ajout des labels pour l'environnement dans la base de données d'entrée
'''
sub_path = ''.join([DB_v1.path, '/2/targets_class.npy'])
if not os.path.exists(sub_path):
    DB_v1.Labelize_random_climate()
else :
    print('Les données sont déjà labelisées')

sub_path = ''.join([DB_v1.path, '/2/targets_paramenv.npy'])
if not os.path.exists(sub_path):
    DB_v1.get_paramenv()
else :
    print('Les paramètres environnement sont déjà isolés dans un fichier')

'''
On vérifie que les fichiers de classe ont bien été chargés
'''
print("Contenu final d'un exemple de la base de données")
DB_v1.Content()



