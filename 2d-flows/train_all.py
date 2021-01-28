import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
from tqdm.notebook import tqdm
from torchsummary import summary
from torchvision import datasets
from torchvision import transforms

from trainer import Trainer

import utils

from utils import softclip
from utils import plot_train_test_reconstructions, plot_train_test_reconstructions_cvae
from models.cnn_sigmaVAE import CNN_sigmaVAE
from models.cnn_sigmacVAE import CNN_sigmacVAE
from models.cnn_sigmaVAE_flow import CNN_sigmaVAE_flow
from models.cnn_sigmacVAE_flow import CNN_sigmacVAE_flow


import evaluation_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model_on_all_datasets(model_type, flow_type, model, num_epochs, learning_rate, window_size, cond_window_size, batch_size, early_stop_patience=20, start_from='1-1', use_validation=False):

    #dataset_1
    machine_names = ['1-1', '1-2','1-3','1-4','1-5','1-6','1-7','1-8','2-1','2-2','2-3','2-4','2-5','2-6','2-7','2-8','2-9','3-1','3-2','3-3','3-4','3-5','3-6','3-7','3-8','3-9','3-10','3-11']

    for machine_name in machine_names:
        print('Training on machine- ' + machine_name)
        done_with_this_server = False

        failed_count=0

        while(not done_with_this_server):

            #VAE

            valloader=None
            if 'cvae' not in model_type:
                if not use_validation:
                    X_train_data, X_test_data, X_train_tensor, X_test_tensor, df_Y_test, trainloader, testloader = utils.read_machine_data('../../datasets/ServerMachineDataset/machine-' + machine_name, window_size, batch_size)
                else:
                    X_train_data, X_test_data, X_train_tensor, X_test_tensor, df_Y_test, trainloader, valloader, testloader = utils.read_machine_data_with_validation('../../datasets/ServerMachineDataset/machine-' + machine_name, window_size, batch_size, val_size=.3)


            else:
                if not use_validation:
                    X_train_data, X_test_data, X_train_tensor, cond_train_tensor, X_test_tensor, cond_test_tensor, df_Y_test, trainloader, testloader = utils.read_machine_data_cvae('../../datasets/ServerMachineDataset/machine-' +machine_name, window_size, cond_window_size, batch_size)
                else:
                    X_train_data, X_test_data, X_train_tensor, cond_train_tensor, X_test_tensor, cond_test_tensor, df_Y_test, trainloader, valloader, testloader = utils.read_machine_data_cvae_with_validation('../../datasets/ServerMachineDataset/machine-' +machine_name, window_size, cond_window_size, batch_size, val_size=.3)

            trainer = Trainer(data_name = machine_name, model_type = model_type, flow_type=flow_type, early_stop_patience=5)
            model, flag = trainer.train_model(model, num_epochs=num_epochs, learning_rate=learning_rate, trainloader=trainloader, valloader=valloader)

            trainer.plot_model_loss()


            if flag:
                return
                #failed
                failed_count+=1
                if failed_count>3:
                    learning_rate /=2
                continue
            else:
                done_with_this_server = True
                torch.save(model, 'saved_models/' + model_type + flow_type + '-' + machine_name + '.pth')



def main():

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type')
    parser.add_argument('--flow_type', nargs='?')
    args = parser.parse_args()
    model_type = args.model_type
    flow_type = args.flow_type

    '''

    model_type='cvae'
    flow_type=None


    print(model_type, flow_type)
    if flow_type == None:
        flow_type=''

    batch_size=256
    latent_dim=10

    if model_type=='vae':
        latent_dim=10
        window_size=32
        cond_window_size=-1

    
        if flow_type=='':
            num_epochs=5
            lr=.005

            model = CNN_sigmaVAE(latent_dim=latent_dim, window_size=window_size).to(device)
            model.cuda() if torch.cuda.is_available() else model.cpu()
            print(model)

        else:
            num_epochs=5
            lr=.002

            model = CNN_sigmaVAE_flow(latent_dim=latent_dim, window_size=window_size, flow_type=flow_type).to(device)
            model.cuda() if torch.cuda.is_available() else model.cpu()
            print(model)
    

    elif model_type=='cvae':	
        window_size=32
        cond_window_size=13
    
        if flow_type=='':
            num_epochs=5
            lr=.005

            model = CNN_sigmacVAE(latent_dim=latent_dim, window_size=window_size, cond_window_size = cond_window_size).to(device)
            model.cuda() if torch.cuda.is_available() else model.cpu()
            print(model)
    
        else:
            num_epochs=5
            lr=.0005

            model = CNN_sigmacVAE_flow(latent_dim=latent_dim, window_size=window_size, cond_window_size=cond_window_size, flow_type=flow_type).to(device)
            model.cuda() if torch.cuda.is_available() else model.cpu()
            print(model)
            
    train_model_on_all_datasets(model_type, flow_type, model, num_epochs, lr, window_size, cond_window_size, batch_size, early_stop_patience=50, start_from='1-1', use_validation=True)
    

if __name__=='__main__':
	main()