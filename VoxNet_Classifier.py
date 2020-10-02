import os
import numpy as np

from Nets.VoxNet_Class import VoxNet
from tensorflow.keras.losses import mean_squared_error, categorical_crossentropy
from tensorflow.keras.optimizers import Adam

def main() :
    from Nets.VoxNet_Class import VoxNet
    from tensorflow.keras.losses import mean_squared_error, categorical_crossentropy
    from tensorflow.keras.optimizers import Adam
    from Visualisation.Visualisation_Matrice_Confusion import draw_heatmap, gif_suivi, main_suivi

    #On Charge le modèle
    VoxNet = VoxNet(n_classes=4)

    #On le compile
    VoxNet.Compile(loss_function=categorical_crossentropy, optimizer = Adam(), metrics=['accuracy'])

    #On regarde le sommaire
    VoxNet.sommaire()

    #On va chercher les données d'entrées
    VoxNet.get_db(DB_name='DB_train', path_to_DB='/home/abel/Desktop/VoxNet_Specie_Sauvegarde/data_specie/')
    
    #print(VoxNet.db.size)

    #On split la DB en deux, pour validation et entrainement
    VoxNet.split_db(test_size=120)

    #On l'entraine. Avec 6*128 exemples entre cahque sauvegarde on reste sur 50% de RAM utilisée. 

    VoxNet.train_and_monitor(path_to_data='/home/abel/Desktop/VoxNet_Specie_Sauvegarde/data_specie/DB_train/',
                      path_model='/home/abel/Desktop/VoxNet_Specie_Sauvegarde/Models/model_specie_classifier.h5',
                      path_to_metrics='/home/abel/Desktop/VoxNet_Specie_Sauvegarde/Metrics_specie/',
                      path_to_weights='//home/abel/Desktop/VoxNet_Specie_Sauvegarde/Weights_specie/',
                      batch_size=960,
                      n_batch_per_step=1,
                      n_epochs=5,
                      n_classes=4)

    main_suivi('/home/abel/Desktop/VoxNet_Specie_Sauvegarde/Metrics_specie/')

main()
