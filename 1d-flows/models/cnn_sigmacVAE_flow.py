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

from maf import MAF, RealNVP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class CNN_sigmacVAE_flow(nn.Module):

    def __init__(self,latent_dim=8, window_size=20, cond_window_size=10, use_probabilistic_decoder=False, flow_type = 'MAF'):
        super(CNN_sigmacVAE_flow, self).__init__()
        
        self.window_size=window_size
        self.cond_window_size=cond_window_size
        self.latent_dim = latent_dim
        self.prob_decoder = use_probabilistic_decoder
        self.flow_type=flow_type
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=6, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=6, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=4, kernel_size=6, stride=1, padding=0)
        self.bn3 = nn.BatchNorm1d(4)

        self.fc41 = nn.Linear(4*15, self.latent_dim)
        self.fc42 = nn.Linear(4*15, self.latent_dim)

        self.defc1 = nn.Linear(self.latent_dim + self.cond_window_size, 4*15)
        
        self.deconv1 = nn.ConvTranspose1d(in_channels=4, out_channels=16, kernel_size=2, stride=1, padding=0, output_padding=0)
        self.debn1 = nn.BatchNorm1d(16)
        self.deconv2 = nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=0, output_padding=0)
        self.debn2 = nn.BatchNorm1d(8)
        self.deconv3 = nn.ConvTranspose1d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=0, output_padding=0)

        self.decoder_fc41 = nn.Linear(window_size, window_size)
        self.decoder_fc42 = nn.Linear(window_size, window_size)
        
        self.log_sigma = torch.nn.Parameter(torch.full((1,), 0.0)[0], requires_grad=True)
        
        if self.flow_type=='RealNVP':
            self.flow = RealNVP(n_blocks=1, input_size=self.latent_dim, hidden_size=50, n_hidden=1)
        
        elif self.flow_type=='MAF':
            self.flow = MAF(n_blocks=1, input_size=self.latent_dim, hidden_size=50, n_hidden=1)
        
        
    def encoder(self, x, c):
        concat_input = torch.cat([x, c], 2)
        h = self.bn1(F.relu(self.conv1(concat_input)))
        h = self.bn2(F.relu(self.conv2(h)))
        h = self.bn3(F.relu(self.conv3(h)))
        
        self.saved_dim = [h.size(1), h.size(2)]
        
        h = h.view(h.size(0), h.size(1) * h.size(2))
        
        return self.fc41(h), self.fc42(h)
    
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu) # return z sample
    
    def decoder(self, z, c):
        c = c.view(c.size(0), -1)
        concat_input = torch.cat([z, c], 1)
        concat_input = self.defc1(concat_input)
        concat_input = concat_input.view(concat_input.size(0), self.saved_dim[0], self.saved_dim[1])

        
        h = self.debn1(F.relu(self.deconv1(concat_input)))
        h = self.debn2(F.relu(self.deconv2(h)))     
        out = torch.sigmoid(self.deconv3(h))
        
        if self.prob_decoder:
            rec_mu = self.decoder_fc41(out).tanh()
            rec_sigma = self.decoder_fc42(out).tanh()
            return out, rec_mu, rec_sigma
        
        #else:
        return out, 0, 0
    
    
    def latent_not_planar(self, x, z_params):
        n_batch = x.size(0)
                
        # Retrieve set of parameters
        #mu, sigma = z_params
        mu, log_var = z_params
        sigma = torch.sqrt(log_var.exp())
        
        # Obtain our first set of latent points
        z0 = self.sampling(mu, log_var)
        
        zk, loss = self.flow.log_prob(z0, None)
        loss = -loss.mean(0)

        return zk, loss
    
 
    def forward(self, x, c):
        
        z_params = self.encoder(x, c)
        mu, log_var = z_params
        
        z_k, kl = self.latent_not_planar(x, z_params)
        
        output, rec_mu, rec_sigma = self.decoder(z_k, c)
   
        return output, rec_mu, rec_sigma, kl
        
        
        
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
