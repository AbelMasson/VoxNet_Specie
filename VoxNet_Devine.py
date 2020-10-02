import os
import numpy as np

from Nets_specie.VoxNet_Devine_Class import VoxNet
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam

def main():
    
    # On Charge le modèle
    VoxNet_ = VoxNet()

    # On le compile
    VoxNet_.Compile(loss_function=mean_squared_error, optimizer=Adam(), metrics=['mean_absolute_percentage_error'])

    # On regarde le sommaire
    VoxNet_.sommaire()

    # On va chercher les données d'entrées
    VoxNet_.get_db(DB_name='DB_train', path_to_DB='/home/abel/Desktop/VoxNet_Specie_Sauvegarde/Data_specie/')
    print(VoxNet_.db.size)

    #On isole les paramètres environnements
    if not os.path.exists('/home/abel/Desktop/VoxNet_Specie_Sauvegarde/Data_specie/DB_train/100/targets_paramenv.npy') :
        VoxNet_.db.get_paramenv()

    # On split la DB en deux, pour validation et entrainement
    VoxNet_.split_db(val_size=120)

    # On l'entraine.

    VoxNet_.train_and_monitor(path_to_data='/home/abel/Desktop/VoxNet_Specie_Sauvegarde/Data_specie/DB_train/',
                             path_model='/home/abel/Desktop/VoxNet_Specie_Sauvegarde/Models_specie/model_specie_devine2.h5',
                             path_to_metrics='/home/abel/Desktop/VoxNet_Specie_Sauvegarde/Metrics_specie/',
                             path_to_weights='/home/abel/Desktop/VoxNet_Specie_Sauvegarde/Weights_specie/',
                             batch_size=960,
                             n_batch_per_step=1,
                             n_epochs=20)


main()
