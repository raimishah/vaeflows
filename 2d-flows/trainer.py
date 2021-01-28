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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(nn.Module):

    def __init__(self, data_name, model_type, early_stop_patience=5):
        super(Trainer, self).__init__()

        self.losses = []
        self.data_name = data_name
        self.model_type = model_type
        self.es = early_stop_patience
        

    def train_model(self, model, num_epochs, learning_rate, trainloader, valloader=None):
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        tq = tqdm(range(num_epochs))

        model.train()

        for epoch in tq:
            flag = False
            for j, data in enumerate(trainloader, 0):

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
                
            if(flag):
                break
            tq.set_postfix(loss=loss.item())

            self.losses.append(loss.item())
                
        return model, flag

    def plot_model_loss(self):
        plt.figure(figsize=(20,6))
        plt.plot(self.losses)
        plt.show()

