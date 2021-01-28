import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

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

from early_stopping import EarlyStopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(nn.Module):

    def __init__(self, data_name, model_type, flow_type='', early_stop_patience=5):
        super(Trainer, self).__init__()

        self.losses = []
        self.val_losses = []
        self.data_name = data_name
        self.model_type = model_type
        self.flow_type = flow_type
        self.es = EarlyStopping(patience=early_stop_patience)

        

    def train_model(self, model, num_epochs, learning_rate, trainloader, valloader=None):
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        tq = tqdm(range(num_epochs))


        early_stopped=False

        for epoch in tq:
            flag = False
            for j, data in enumerate(trainloader, 0):
                model.train()

                optimizer.zero_grad()

                #batches
                x, y = data
                x = x.cuda() if torch.cuda.is_available() else x.cpu()
                x.to(device)
                y = y.cuda() if torch.cuda.is_available() else y.cpu()
                y.to(device)

                if self.model_type=='cvae':
                    outputs, rec_mu, rec_sigma, kl = model(x, y)
                else:
                    outputs, rec_mu, rec_sigma, kl = model(x)

                rec_comps, rec, rec_mu_sigma_loss, kl = model.loss_function(outputs, x, rec_mu, rec_sigma, kl)

                loss = rec + kl + rec_mu_sigma_loss

                if(np.isnan(loss.item())):
                    print("Noped out at", epoch, j, kl, rec_comps)
                    flag = True
                    break

                loss.backward()
                optimizer.step()


            if valloader != None:
                #VALIDATION
                with torch.no_grad():
                    model.eval()
                    for j, data in enumerate(valloader, 0):
                        x, y = data
                        x = x.cuda() if torch.cuda.is_available() else x.cpu()
                        x.to(device)
                        y = y.cuda() if torch.cuda.is_available() else y.cpu()
                        y.to(device)
                        if self.model_type=='cvae':
                            outputs, rec_mu, rec_sigma, kl = model(x, y)
                        else:
                            outputs, rec_mu, rec_sigma, kl = model(x)
                        _, rec, _, _ = model.loss_function(outputs, x, rec_mu, rec_sigma, kl)

                        val_loss = rec 
                        if(np.isnan(val_loss.item())):
                            print("Noped out in validation at", epoch, j, kl, rec_comps)
                            flag = True
                            break

                        if self.es.step(val_loss):
                            early_stopped=True
                            break
                
            if(flag) or early_stopped:
                break
            tq.set_postfix(loss=loss.item())

            self.losses.append(loss.item())
            self.val_losses.append(val_loss.item())
                
        return model, flag

    def plot_model_loss(self):
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,6))
        ax1.plot(self.losses,label='loss (total)', color='blue')
        ax2.plot(self.val_losses,label='validation loss (reconstruction only (MSE))', color='orange')
        plt.legend()
        #plt.savefig('saved_models/' + self.model_type + self.flow_type + '-' + self.data_name + '.png')
        plt.show()

