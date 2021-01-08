import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.autograd import Variable

from torchsummary import summary

import torchvision
from torchvision import datasets
from torchvision import transforms

from utils import softclip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN_sigmaVAE(nn.Module):

    def __init__(self,latent_dim=8, window_size=20, use_probabilistic_decoder=False):
        super(CNN_sigmaVAE, self).__init__()
        
        self.window_size=window_size
        self.latent_dim = latent_dim
        self.prob_decoder = use_probabilistic_decoder
        
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=4, kernel_size=5, stride=1, padding=0)

        #self.fc41 = nn.Linear(4*8, self.latent_dim)
        #self.fc42 = nn.Linear(4*8, self.latent_dim)
        #self.defc1 = nn.Linear(self.latent_dim, 4*8)
        
        self.fc41 = nn.Linear(4*116, self.latent_dim)
        self.fc42 = nn.Linear(4*116, self.latent_dim)
        self.defc1 = nn.Linear(self.latent_dim, 4*116)
        
        
        self.deconv1 = nn.ConvTranspose1d(in_channels=4, out_channels=16, kernel_size=5, stride=1, padding=0, output_padding=0)
        self.deconv2 = nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=5, stride=1, padding=0, output_padding=0)
        self.deconv3 = nn.ConvTranspose1d(in_channels=8, out_channels=1, kernel_size=5, stride=1, padding=0, output_padding=0)

        self.log_sigma = 0
        self.log_sigma = torch.nn.Parameter(torch.full((1,), 0.0)[0], requires_grad=True)
        
        
        self.decoder_fc41 = nn.Linear(self.window_size, self.window_size)
        self.decoder_fc42 = nn.Linear(self.window_size, self.window_size)
        
        self.decoder_fc43 = nn.Linear(self.window_size, self.window_size)
        self.decoder_fc44 = nn.Linear(self.window_size, self.window_size)
        
        
    def encoder(self, x):
        concat_input = x #torch.cat([x, c], 1)
        h = F.relu(self.conv1(concat_input))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))

        
        self.saved_dim = [h.size(1), h.size(2)]
        
        h = h.view(h.size(0), h.size(1) * h.size(2))
        
        return self.fc41(h), self.fc42(h)
    
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu) # return z sample
    
    def decoder(self, z):
        concat_input = z #torch.cat([z, c], 1)
        concat_input = self.defc1(concat_input)
        concat_input = concat_input.view(concat_input.size(0), self.saved_dim[0], self.saved_dim[1])
        
        h = F.relu(self.deconv1(concat_input))
        h = F.relu(self.deconv2(h))
        
        out = torch.sigmoid(self.deconv3(h))
        
        if self.prob_decoder:
            rec_mu = self.decoder_fc43(out).tanh()
            rec_sigma = self.decoder_fc44(out).tanh()
            return out, rec_mu, rec_sigma
        
        else:
            return out, 0, 0
    
    def forward(self, x):

        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        output, rec_mu, rec_sigma = self.decoder(z)

        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return output, rec_mu, rec_sigma, kl_div


    def gaussian_nll(self, mu, log_sigma, x):
        return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)

    
    def reconstruction_loss(self, x_hat, x):

        log_sigma = self.log_sigma
        log_sigma = softclip(log_sigma, -6)
        
        rec_comps = self.gaussian_nll(x_hat, log_sigma, x)
        rec = rec_comps.sum()

        return rec_comps, rec

    
    def loss_function(self, recon_x, x, rec_mu, rec_sigma, kl):
        
        rec_comps, rec = self.reconstruction_loss(recon_x, x)
        #kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        rec_mu_sigma_loss = 0
        if self.prob_decoder:
            rec_mu_sigma_loss = self.gaussian_nll(rec_mu, rec_sigma, x).sum()
        
        return rec_comps, rec, rec_mu_sigma_loss, kl