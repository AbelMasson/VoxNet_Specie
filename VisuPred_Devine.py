#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 17:37:54 2020

@author: lenovo
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import glob
import imageio
from pathlib import Path
from sklearn.metrics import mean_squared_error

class Visualisation_VoxNet_Devine(object):
    
    def __init__(self, path_to_results, y_pred_name, y_test_name, val_loss_name, train_loss_name, metrics_name, n_epochs) :
        
        self.path_to_results = path_to_results
        self.y_pred_name = y_pred_name
        self.y_test_name = y_test_name
        self.val_loss_name = val_loss_name
        self.train_loss_name = train_loss_name
        self.metrics_name = metrics_name
        self.n_epochs = n_epochs
        
        self.dict = {}
        self.y_test = np.load(os.path.join(self.path_to_results, self.y_test_name))
        self.val_loss = np.load(os.path.join(self.path_to_results, self.val_loss_name))
        self.train_loss_raw = np.load(os.path.join(self.path_to_results, self.train_loss_name))
        train_loss = []
        for i in range(len(self.train_loss_raw)):
            for j in range(len(self.train_loss_raw[0])):
                train_loss.append(self.train_loss_raw[i][j])
        self.train_loss = train_loss
        self.metrics_raw = np.load(os.path.join(self.path_to_results, self.metrics_name))
        metrics = []
        for i in range(len(self.metrics_raw)):
            for j in range(len(self.metrics_raw[0])):
                metrics.append(self.metrics_raw[i][j])
        self.metrics = metrics
        for i in range(n_epochs) :
            self.dict['y_pred_epoch{}'.format(i)] = np.load(os.path.join(self.path_to_results, self.y_pred_name.format(i)))
            
    def trace_val_loss(self):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(self.val_loss[0:17])
        ax.set_xticks([0,5,10,15,17])
        ax.set_xticklabels([0,5,10,15,17], fontsize=8)
        ax.set_yticks([0.0,0.01,0.02])
        ax.set_yticklabels([0.0,0.01,0.02], fontsize=8)
        ax.set_xlabel('epoch', fontweight='bold')
        ax.set_ylabel('loss : RMSE', fontweight='bold')
        #ax.title("Evolution de la loss (RMSE) de validation en fonction du nombre d'epochs")
        #fig.show()
        fig_name = 'Evolution_val_lossf.png'
        plt.savefig(os.path.join(self.path_to_results, fig_name))
        
    def trace_train_loss(self):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        self.n_batch_per_step = 1
        self.n_epoch_per_batch = 5
        self.n_step_per_epoch = 15
        ax.plot(list(self.train_loss[0:17*self.n_batch_per_step*self.n_epoch_per_batch*self.n_step_per_epoch]))
        ax.set_xticks([i**self.n_batch_per_step*self.n_epoch_per_batch*self.n_step_per_epoch for i in [0,5,10,15,17]])
        ax.set_xticklabels([0,5,10,15,17])
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss : RMSE')
        plt.title("Evolution de la loss (RMSE) d'entrainement en fonction du nombre d'epochs")
        #fig.show()
        fig_name = 'Evolution_train_lossf.png'
        plt.savefig(os.path.join(self.path_to_results, fig_name))
    
    def trace_train_losses(self):
        fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(14,3))
        self.n_batch_per_step = 1
        self.n_epoch_per_batch = 5
        self.n_step_per_epoch = 15
        
        ax[0].plot(list(self.train_loss[0*self.n_batch_per_step*self.n_epoch_per_batch*self.n_step_per_epoch:(0+1)*self.n_batch_per_step*self.n_epoch_per_batch*self.n_step_per_epoch]))
        
        ax[0].set_xticks([0,37,74])
        ax[0].set_xticklabels([0,50,100], fontsize=8)
        ax[0].set_yticks([0,0.15])
        ax[0].set_yticklabels([0,0.15], fontweight='bold', fontsize=8)
        #ax[0].set_xlabel("Pourcentage parcouru de la base d'entrainement", fontweight='bold', fontsize=8)
        ax[0].set_ylabel('loss : RMSE', fontweight='bold', fontsize=8)
        ax[0].set_title('epoch 0', fontstyle='italic', fontsize=8)
        #plt.title("Evolution de la loss (RMSE) d'entrainement en fonction du nombre d'epochs")
        #fig.show()
        
        for i in range(1,8) :
            ax[1].plot(list(self.train_loss[i*self.n_batch_per_step*self.n_epoch_per_batch*self.n_step_per_epoch:(i+1)*self.n_batch_per_step*self.n_epoch_per_batch*self.n_step_per_epoch]))
        
        ax[1].set_xticks([0,37,74])
        ax[1].set_xticklabels([0,50,100], fontsize=8)
        ax[1].set_yticks([0,0.03])
        ax[1].set_yticklabels([0, 0.03], fontweight='bold', fontsize=8)
        #ax[1].set_xlabel("Pourcentage parcouru de la base d'entrainement", fontweight='bold')
        #ax[1].set_ylabel('loss : RMSE', fontweight='bold')
        ax[1].set_title('epochs 1 à 7', fontstyle='italic', fontsize=8)
        #plt.title("Evolution de la loss (RMSE) d'entrainement en fonction du nombre d'epochs")
        #fig.show()
        
        for i in range(8,13) :
            ax[2].plot(list(self.train_loss[i*self.n_batch_per_step*self.n_epoch_per_batch*self.n_step_per_epoch:(i+1)*self.n_batch_per_step*self.n_epoch_per_batch*self.n_step_per_epoch]))
        
        ax[2].set_xticks([0,37,74])
        ax[2].set_xticklabels([0,50,100], fontsize=8)
        ax[2].set_yticks([0,0.003])
        ax[2].set_yticklabels([0,0.003], fontweight='bold', fontsize=8)
        #ax[2].set_xlabel("Pourcentage parcouru de la base d'entrainement", fontweight='bold')
        #ax[2].set_ylabel('loss : RMSE', fontweight='bold')
        ax[2].set_title('epochs 8 à 13', fontstyle='italic', fontsize=8)
        #plt.title("Evolution de la loss (RMSE) d'entrainement en fonction du nombre d'epochs")
        #fig.show()
        
        for i in [13,14,16,17] :
            ax[3].plot(list(self.train_loss[i*self.n_batch_per_step*self.n_epoch_per_batch*self.n_step_per_epoch:(i+1)*self.n_batch_per_step*self.n_epoch_per_batch*self.n_step_per_epoch]))
        
        ax[3].set_xticks([0,37,74])
        ax[3].set_xticklabels([0,50,100], fontsize=8)
        ax[3].set_yticks([0,0.001])
        ax[3].set_yticklabels([0,0.001], fontweight='bold', fontsize=8)
        #ax[2].set_xlabel("Pourcentage parcouru de la base d'entrainement", fontweight='bold')
        #ax[2].set_ylabel('loss : RMSE', fontweight='bold')
        ax[3].set_title('epochs 13 à 17', fontstyle='italic', fontsize=8)
        #plt.title("Evolution de la loss (RMSE) d'entrainement en fonction du nombre d'epochs")
        #fig.show()
        
        fig_name = 'Evolution_train_lossesf.png'
        plt.savefig(os.path.join(self.path_to_results, fig_name))
        
    def trace_metrics(self):
        fig = plt.figure()
        x = [i*20/len(self.metrics) for i in range(len(self.metrics))]
        plt.plot(x, list(self.metrics))
        plt.title("Evolution du pourcentage moyen de l'erreur absolue en fonction du nombre d'epochs")
        #fig.show()
        fig_name = 'Evolution_metrics2.png'
        plt.savefig(os.path.join(self.path_to_results, fig_name))
    
    def split_water_light(self) :
        
        self.dict_test = {}
        
        self.dict_test['Water'] = [self.y_test[i][0] for i in range(len(self.y_test))]
        self.dict_test['Light'] = [self.y_test[i][1] for i in range(len(self.y_test))]

        
        for epoch in range(self.n_epochs) :
        
            y_pred = self.dict['y_pred_epoch{}'.format(epoch)]
            
            self.dict['y_pred_epoch{}_Water'.format(epoch)] = [y_pred[i][0] for i in range(len(y_pred))]
            self.dict['y_pred_epoch{}_Light'.format(epoch)] = [y_pred[i][1] for i in range(len(y_pred))]
        
    def process(self, epoch, paramenv) :
        
        y_pred_name = 'y_pred_epoch{}'.format(epoch) + '_' + str(paramenv)
        
        y_pred = list(np.copy(self.dict[y_pred_name]))
        y_test = list(np.copy(self.dict_test[paramenv]))
        
        y_test_new = []; y_pred_new = []
        
        for i in range(len(y_test)) :
            
            index = y_test.index(min(y_test))
            
            y_test_new.append(y_test[index])
            y_pred_new.append(y_pred[index])
            
            y_test.remove(y_test[index])
            y_pred.remove(y_pred[index])
            #print(self.dict[y_pred_name])
        
        #print(y_test_new, y_pred_new)
        return y_test_new, y_pred_new
        
    def trace_pred(self, axes, ax_i, epoch) :
        
        y_test_Water, y_pred_Water = self.process(epoch, 'Water')
        #print(y_test_water, y_pred_water)
        y_test_Light, y_pred_Light = self.process(epoch, 'Light')
        #print(y_test_light, y_pred_light)
        
        RMSE_Sm = mean_squared_error(y_test_Water, y_pred_Water)
        RMSE_Sm = round(RMSE_Sm, 5)
        RMSE_Sr = mean_squared_error(y_test_Light, y_pred_Light)
        RMSE_Sr = round(RMSE_Sr, 5)

        #props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        #ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        #verticalalignment='top', bbox=props)
        x = [i for i in range(len(y_test_Water))]
        
        axes[ax_i, 0].plot(y_test_Water, label='y_test')
        axes[ax_i, 0].scatter(x, y_pred_Water, s=8, c='darkred', label='y_pred')
        axes[ax_i, 0].set_title('prediction Water', fontstyle='italic', fontsize=8)
        axes[ax_i, 0].set_yticks([0.5,1])
        axes[ax_i, 0].set_yticklabels([0.5,1], fontsize=7)
        axes[ax_i, 0].set_xticks([0,len(y_test_Water)])
        axes[ax_i, 0].set_xticklabels([0,len(y_test_Water)], fontsize=8)
        #axes[ax_i, 0].text(20, 0.05, 'RMSE = '+str(RMSE_Sm), fontsize=8, fontweight='bold')
        
        axes[ax_i, 1].plot(y_test_Light, label='y_test')
        axes[ax_i, 1].scatter(x, y_pred_Light, s=8, c='darkred', label='y_pred')
        axes[ax_i, 1].set_title('prediction Light', fontstyle='italic', fontsize=8)
        axes[ax_i, 1].set_yticks([0.5,1])
        axes[ax_i, 1].set_yticklabels([0.5,1], fontsize=7)
        axes[ax_i, 1].set_xticks([0,len(y_test_Light)])
        axes[ax_i, 1].set_xticklabels([0,len(y_test_Light)], fontsize=8)
        #axes[ax_i, 1].text(20, 0.075, 'RMSE = '+str(RMSE_Sr), fontsize=8, fontweight='bold')

        axes[ax_i, 1].legend(fontsize=8)
    
    def trace_hist(self, axes, ax_i, epoch) :
        y_test_Water, y_pred_Water = self.process(epoch, 'Water')
        #print(y_test_water, y_pred_water)
        y_test_Light, y_pred_Light = self.process(epoch, 'Light')
        #print(y_test_light, y_pred_light)
        
        d_Sm = [abs(y_test_Water[i] - y_pred_Water[i])*100/y_test_Water[i] for i in range(len(y_test_Water))]
        d_Sm2 = [(y_test_Water[i] - y_pred_Water[i])*100/y_test_Water[i] for i in range(len(y_test_Water))]
        mu_Sm = np.mean(d_Sm)
        med_Sm = np.median(d_Sm)
        s_Sm = np.asarray(d_Sm2).std()
        d_Sr = [abs(y_test_Light[i] - y_pred_Light[i])*100/y_test_Light[i] for i in range(len(y_test_Light))]
        d_Sr2 = [(y_test_Light[i] - y_pred_Light[i])*100/y_test_Light[i] for i in range(len(y_test_Light))]
        mu_Sr = np.mean(d_Sr)
        s_Sr = np.asarray(d_Sr).std()
        med_Sr = np.median(d_Sr)
        
        #sns.distplot(d_Sm, ax = axes[ax_i, 0])
        axes[ax_i, 0].hist(d_Sm, bins=[i for i in range(0,100,2)])
        axes[ax_i, 0].axvline(x=med_Sm, ymin=0, ymax=20, color='darkred')
        axes[ax_i, 0].set_xticks([med_Sm])
        axes[ax_i, 0].set_xticklabels([round(med_Sm,1)], fontsize=8, fontweight='bold', color='darkred')
        axes[ax_i, 0].set_yticks([0,10,50])
        axes[ax_i, 0].set_yticklabels([0,10,50], fontsize=8)
        textstr =  '\n'.join((r'$\mu=%.2f$'%(mu_Sm, ), r'$\sigma=%.2f$'%(s_Sm, )))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        axes[ax_i, 0].text(mu_Sm+20, 15, textstr, fontsize=6, fontweight='bold', bbox=props)
        axes[ax_i, 0].set_title('RApE Water', fontstyle='italic', fontsize=8)
        
        #sns.distplot(d_Sr, ax = axes[ax_i, 1])
        axes[ax_i, 1].hist(d_Sr, bins=[i for i in range(0,100,2)])
        axes[ax_i, 1].axvline(x=med_Sr, ymin=0, ymax=30, color='darkred')
        axes[ax_i, 1].set_xticks([med_Sr])
        axes[ax_i, 1].set_xticklabels([round(med_Sr,1)], fontsize=8, fontweight='bold',  color='darkred')
        axes[ax_i, 1].set_yticks([0,10,50,75])
        axes[ax_i, 1].set_yticklabels([0,10,50,75], fontsize=8)
        textstr =  '\n'.join((r'$\mu=%.2f$'%(mu_Sr, ), r'$\sigma=%.2f$'%(s_Sr, )))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        axes[ax_i, 1].text(mu_Sr+20, 20, textstr, fontsize=6, fontweight='bold', bbox=props)
        axes[ax_i, 1].set_title('RApE Light', fontstyle='italic', fontsize=8)
        
    def trace_main(self, epoch) :
        
        fig, axes = plt.subplots(nrows=2, ncols=2)
        self.trace_pred(axes, 0, epoch)
        self.trace_hist(axes, 1, epoch)
        plt.suptitle('Prediction epoch {}'.format(epoch), fontsize='large', fontweight='bold')
        fig_name = ''.join([self.y_pred_name[0:-4].format(epoch), '.png'])
        plt.savefig(os.path.join(self.path_to_results, fig_name))
        
    def animate_pred(self) :
        
        image_list = []
        for epoch in range(self.n_epochs) :
            
            image_name = ''.join([self.y_pred_name[0:-4].format(epoch), '.png'])
            image_path = os.path.join(self.path_to_results, image_name)
            image_list.append(imageio.imread(image_path))
            
        imageio.mimwrite(''.join([self.y_pred_name[0:-4].format(epoch), '.gif']), image_list, loop=1, fps=5)
    
    def trace_all_pred(self) :
        
        for epoch in range(self.n_epochs) :
            self.trace_main(epoch)
        
        self.animate_pred()

        
    def animate_pred(self) :
        
        image_list = []
        self.n_epochs = 18
        for epoch in range(self.n_epochs) :
            
            image_name = 'Prediction2_epoch{}.png'.format(epoch)
            image_path = os.path.join(self.path_to_results, image_name)
            image_list.append(imageio.imread(image_path))
            
        imageio.mimwrite('animated_prediction2.gif', image_list, loop=2, fps=3)

if __name__ == '__main__' :
    
    Visu = Visualisation_VoxNet_Devine(path_to_results = '/home/abel/Desktop/VoxNet_Specie_Sauvegarde/Metrics_specie/',
                                       y_pred_name = 'y_val_pred2_epoch_{}.npy', 
                                       y_test_name = 'y_val2.npy', 
                                       val_loss_name = 'val_loss_per_epoch2.npy',
                                       train_loss_name = 'train_loss2.npy',
                                       metrics_name = 'train_metrics2.npy', 
                                       n_epochs = 18)
    
    Visu.trace_val_loss()
    Visu.trace_train_loss()
    Visu.trace_train_losses()
    #Visu.trace_metrics()
    #Visu.split_water_light()
    #Visu.trace_all_pred()
    #Visu.animate_pred()
        
        
        
