import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
import torch.distributions as distrib
import torch.distributions.transforms as transform

from torchsummary import summary

import torchvision
from torchvision import datasets
from torchvision import transforms

from utils import softclip

from maf import MAF, RealNVP
from planar_flow import PlanarFlow, NormalizingFlow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN_sigmaVAE_flow(nn.Module):

    def __init__(self,latent_dim=8, window_size=20, use_probabilistic_decoder=False, flow_type = 'MAF'):
        super(CNN_sigmaVAE_flow, self).__init__()
    
        self.window_size=window_size
        self.latent_dim = latent_dim
        self.prob_decoder = use_probabilistic_decoder
        self.flow_type=flow_type
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=5, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(4)

        self.fc41 = nn.Linear(4*20*26, self.latent_dim)
        self.fc42 = nn.Linear(4*20*26, self.latent_dim)

        self.defc1 = nn.Linear(self.latent_dim, 4*20*26)
        
        self.deconv1 = nn.ConvTranspose2d(in_channels=4, out_channels=16, kernel_size=5, stride=1, padding=0, output_padding=0)
        self.debn1 = nn.BatchNorm2d(16)
        self.deconv2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=5, stride=1, padding=0, output_padding=0)
        self.debn2 = nn.BatchNorm2d(8)
        self.deconv3 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=5, stride=1, padding=0, output_padding=0)

        self.log_sigma = 0
        self.log_sigma = torch.nn.Parameter(torch.full((1,), 0.0)[0], requires_grad=True)
        
        self.decoder_fc41 = nn.Linear(self.window_size, self.window_size)
        self.decoder_fc42 = nn.Linear(self.window_size, self.window_size)

        if self.flow_type=='RealNVP':
            self.flow = RealNVP(n_blocks=1, input_size=self.latent_dim, hidden_size=50, n_hidden=1)
        
        elif self.flow_type=='MAF':
            self.flow = MAF(n_blocks=1, input_size=self.latent_dim, hidden_size=50, n_hidden=1)
        
        elif self.flow_type =='Planar':
            self.block_planar = [PlanarFlow]
            self.flow = NormalizingFlow(dim=self.latent_dim, blocks=self.block_planar, flow_length=16, density=distrib.MultivariateNormal(torch.zeros(self.latent_dim), torch.eye(self.latent_dim)))



        
    def encoder(self, x):
        
        concat_input = x #torch.cat([x, c], 2)
        
        h = F.relu(self.conv1(concat_input))
        h = self.bn2(F.relu(self.conv2(h)))
        h = self.bn3(F.relu(self.conv3(h)))
        
        self.saved_dim = [h.size(1), h.size(2), h.size(3)]
        

        h = h.view(h.size(0), -1)
                
        mu, var = self.fc41(h), self.fc42(h)
                
        return self.fc41(h), self.fc42(h)

    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu) # return z sample
    
    def decoder(self, z):
        
        concat_input = z #torch.cat([z, c], 2)
        concat_input = self.defc1(concat_input)
        concat_input = concat_input.view(concat_input.size(0), self.saved_dim[0], self.saved_dim[1], self.saved_dim[2])


        h = self.debn1(F.relu(self.deconv1(concat_input)))
        h = self.debn2(F.relu(self.deconv2(h)))
        
        out = torch.sigmoid(self.deconv3(h))
        
        if self.prob_decoder:
            rec_mu = self.decoder_fc43(out).tanh()
            rec_sigma = self.decoder_fc44(out).tanh()
            return out, rec_mu, rec_sigma
        
        else:
            return out, 0, 0



    def latent_planar(self, x, z_params):
        
        n_batch = x.size(0)
        
        # Retrieve set of parameters
        #mu, sigma = z_params
        mu, log_var = z_params
        sigma = torch.sqrt(log_var.exp())
        
        # Re-parametrize a Normal distribution
        if torch.cuda.is_available():
            q = distrib.Normal(torch.zeros(mu.shape[1]).cuda(), torch.ones(sigma.shape[1]).cuda())
        else:
            q = distrib.Normal(torch.zeros(mu.shape[1]), torch.ones(sigma.shape[1]))
            
        # Obtain our first set of latent points
        z_0 = (sigma * q.sample((n_batch, ))) + mu
        # Complexify posterior with flows
        z_k, SLDJ = self.flow(z_0)        
        # ln p(z_k) 
        log_p_zk = -0.5 * z_k * z_k
        # ln q(z_0)
        log_q_z0 = -0.5 * (sigma.log() + (z_0 - mu) * (z_0 - mu) * sigma.reciprocal())
        #  ln q(z_0) - ln p(z_k)
        logs = (log_q_z0 - log_p_zk).sum()
        
        # Add log determinants
        ladj = torch.cat(SLDJ)
        # ln q(z_0) - ln p(z_k) - sum[log det]
        logs -= torch.sum(ladj)
        
        #logs -= SLDJ.sum()
        
        return z_k, (logs / float(n_batch))


    
    
    
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
        
    
    def forward(self, x):

        z_params = self.encoder(x)
        mu, log_var = z_params
        
        if self.flow_type =='Planar':
            z_k, kl = self.latent_planar(x, z_params)
            
        else:
            z_k, kl = self.latent_not_planar(x, z_params)
        
        output, rec_mu, rec_sigma = self.decoder(z_k)
   
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
        rec_mu_sigma_loss = 0
        if self.prob_decoder:
            rec_mu_sigma_loss = self.gaussian_nll(rec_mu, rec_sigma, x).sum()
        
        return rec_comps, rec, rec_mu_sigma_loss, kl