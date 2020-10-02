#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 10:23:09 2020

@author: abel

Description : Ce script définit les fonctions nécessaires aux conversions des fichiers .vct d'images 3D et .csv de paramètres en
fichiers .npy.
"""
import os
import struct
import numpy as np
import pandas as pd

def NormTrans(voxel_value, max_value) :
    if max_value != 0 :
        return -1 + 2*(voxel_value/max_value)
    return -1

def convert_vct_to_npy(path_image_file, path_image_to_save, vct_image_name='imgabswood.vct', npy_image_name='imgabswood.npy'):
    '''
    :param path_image_file: Emplacement du fichier contenant l'image 3D
    :param vct_image_name: Nom du fichier contenant l'image 3D. (Par défaut le nom de ce fichier est 'imgabswood.vct')
    :param path_image_to_save: Emplacement où sauvegarder le fichier .npy contenant l'image 3D.
    :param npy_image_name: Nom du fichier .npy à sauvegarder. (Par défaut ce nom est 'imgabswood.npy')
    :return: Cette fonction charge, convertit et change la structure de l'image 3D contenue dans un fichier .vct,
    d'une liste de 27000 flottants à un tableau numpy de dimension 30*30*30, dont les coefficients sont compris
    entre -1 et 1. Elle sauvegarde ce tableau au format .npy à l'emplacement choisi.
    '''

    #Lecture du fichier .vct et conversion en liste
    path_to_file = os.path.join(path_image_file, vct_image_name)
    file = open(path_to_file, 'rb')

    s = struct.Struct("<ffffffffffffffffffffffffffffff")
    ls = []

    while True:
        record = file.read(120)
        if len(record) != 120:
            break;
        ls += list(s.unpack(record))

    # Normalisation de la liste, en séparant la liste_abs de la liste_bois
    # Puis transformation pour que la valeur de chaque voxel soit comprise entre -1 et 1

    max_abs=max(ls[0:13500]); max_bois=max(ls[13500:27000])
    l_abs = [NormTrans(x, max_abs) for x in ls[0:13500]]
    l_bois = [NormTrans(x, max_bois) for x in ls[13500:27000]]

    data = np.asarray(l_abs + l_bois)
    data = np.reshape(data, (30,30,30))
    path_to_npy = os.path.join(path_image_to_save, npy_image_name)
    np.save(path_to_npy, data)

def convert_csv_to_npy(path_target_file, path_target_to_save, csv_target_name='targets.csv', npy_target_name='targets.npy'):
    '''
    :param path_target_file: Emplacement du fichier .csv contenant les paramètres de TOY
    :param csv_target_name: Nom du fichier .csv contenant les paramètres de TOY
    (par défaut ce nom est 'targets.csv')
    :param path_target_to_save: Emplacement où sauvegarder le fichier .npy contenant les paramètres de TOY
    :param npy_target_name: Nom sous lequel sauvegarder le fichier .npy contenant les paramètres de TOY
    (par défaut ce nom est 'targets.npy')
    :return: Cette fonction charge, et convertit un fichier .csv contenant les paramètres de TOY en fichier
    .npy puis sauvegarde ce fichier à l'emplacement choisi (généralement la base de données d'entrée du réseau)
    le fichier .npy résultant contient la liste des paramètres sous forme de tableau numpy à une dimension.
    '''

    l_output = []
    parameter_file = path_target_file + csv_target_name

    df_parameters = pd.read_csv(parameter_file, header=None, sep=' ').T

    # water paramter
    l_output.append(float(df_parameters[0][1]))
    # light parameter (normalisé, ie divisé par valeur max de 200)
    l_output.append(float(df_parameters[1][1]) / 200)
    # shoot1 parameter Tronc
    l_output.append(float(df_parameters[2][1]))
    l_output.append(float(df_parameters[2][2]))
    l_output.append(float(df_parameters[2][3]))
    # shoot2 parameter Rameaux longs
    l_output.append(float(df_parameters[3][1]))
    l_output.append(float(df_parameters[3][2]))
    l_output.append(float(df_parameters[3][3]))
    # shoot3 parameter Rameaux courts (1 seul paramètre)
    l_output.append(float(df_parameters[4][1]))
    # root1 parameter Racines structurantes
    l_output.append(float(df_parameters[5][1]))
    l_output.append(float(df_parameters[5][2]))
    l_output.append(float(df_parameters[5][3]))
    # root2 parameter Racines secondaires
    l_output.append(float(df_parameters[6][1]))
    l_output.append(float(df_parameters[6][2]))
    l_output.append(float(df_parameters[6][3]))
    # root3 parameter Racines absorbantes (1 seul paramètre)
    l_output.append(float(df_parameters[7][1]))

    parameter_output = np.asarray(l_output)
    parameter_output_file = os.path.join(path_target_to_save, npy_target_name)

    np.save(parameter_output_file, parameter_output)

def create_label_env(path_npy_target, npy_target_name, VoxNet_class_dictionary, class_name='targets_class.npy'):
    '''
    Les classes sont distribuées en fonction des ConditionsEau et des ConditionsLumière
    Différentes valeurs seuils pour l'eau et la lumière sont définis de la façon suivante :

    Pour l'eau : HS (Hydric Stress) : W=0.4
             LW (Low Water) : W=0.6
             IW (Intermediate Water) : W=0.8
             HW (High Water) : W=1
    Pour la lumière : LL (Low-Light) : L=0.5
                  IL (Intermediate Light) : L=0.75
                  HL (High Light) : L=1

    Elles permettent de définir 12 classes de conditions environnementales, répertoriées dans
    le dictionnaire ci dessous :

    VoxNet_class_dictionary = { (0.4, 0.5): 0, (0.4, 0.75): 1, (0.4, 1): 2, (0.6, 0.5): 3,
                              (0.6, 0.75): 4, (0.6, 1): 5, (0.8, 0.5): 6, (0.8, 0.75): 7,
                              (0.8, 1): 8, (1, 0.5): 9, (1, 0.75): 10, (1, 1): 11 }

    :param path_npy_target: Emplacement du fichier .npy contenant les paramètres de TOY
    :param npy_target_name: Nom du fichier .npy contenant les parmètres de TOY
    :param VoxNet_class_dictionary: Dictionnaire pour la classification par conditions environnementales
    :param class_name: Nom du fichier .npy contenant l'entier décrivant la classe de l'exemple en question
    :return: Cette fonction détermine la classe (entier entre 0 et 11) de l'exemple étudié en fonction des
    valeurs des paramètres de TOY contenues dans le fichier targets.npy, puis sauvegarde
    cette information dans un fichier .npy à l'emplacement du fichier targets.npy
    '''

    parameters = np.load(os.path.join(path_npy_target,npy_target_name))

    class_file = VoxNet_class_dictionary[(float(parameters[0]), float(parameters[1]))]
    path_class_file = os.path.join(path_npy_target, class_name)

    np.save(path_class_file, class_file)

def create_label_random(path_npy_target, npy_target_name, class_name='targets_class.npy') :

        parameters = np.load(os.path.join(path_npy_target, npy_target_name))
        water = float(parameters[0])
        light = float(parameters[1])
        
        if water > 0.75 :
            if light > 0.75 :
                class_file = 4
            if light <= 0.75 :
                class_file = 3
        if water <= 0.75 :
            if light > 0.75 :
                class_file = 2
            if light <= 0.75 :
                class_file = 1

        path_class_file = os.path.join(path_npy_target, class_name)

        np.save(path_class_file, class_file)

def isolate_paramenv(path_npy_target, npy_target_name, paramenv_name='targets_paramenv.npy') :
    parameters = np.load(os.path.join(path_npy_target, npy_target_name))
    paramenv = np.asarray([float(parameters[0]), float(parameters[1])])

    np.save(os.path.join(path_npy_target, paramenv_name), paramenv)

