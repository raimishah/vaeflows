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
from bnaf import Tanh, MaskedWeight, BNAF, Sequential, Permutation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
class CNN_sigmaVAE(nn.Module):

    def __init__(self,latent_dim=8, window_size=20, use_probabilistic_decoder=False):
        super(CNN_sigmaVAE, self).__init__()
    
        self.window_size=window_size
        self.latent_dim = latent_dim
        self.prob_decoder = use_probabilistic_decoder
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(15,5), stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(10,5), stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(7,3), stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(7,3), stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=(5,3), stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(4)

        self.fc41 = nn.Linear(4*61*24, self.latent_dim)
        self.fc42 = nn.Linear(4*61*24, self.latent_dim)

        self.defc1 = nn.Linear(self.latent_dim, 4*61*24)
        
        self.deconv1 = nn.ConvTranspose2d(in_channels=4, out_channels=16, kernel_size=(5,3), stride=1, padding=0, output_padding=0)
        self.debn1 = nn.BatchNorm2d(16)
        self.deconv2 = nn.ConvTranspose2d(in_channels=16, out_channels=64, kernel_size=(7,3), stride=1, padding=0, output_padding=0)
        self.debn2 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(7,3), stride=1, padding=0, output_padding=0)
        self.debn3 = nn.BatchNorm2d(32)
        self.deconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(10,5), stride=1, padding=0, output_padding=0)
        self.debn4 = nn.BatchNorm2d(8)
        self.deconv5 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(15,5), stride=1, padding=0, output_padding=0)

        self.log_sigma = 0
        self.log_sigma = torch.nn.Parameter(torch.full((1,), 0.0)[0], requires_grad=True)
        
        self.decoder_fc41 = nn.Linear(self.window_size, self.window_size)
        self.decoder_fc42 = nn.Linear(self.window_size, self.window_size)

        
        
    def encoder(self, x):
        
        concat_input = x #torch.cat([x, c], 2)
        
        
        h = F.relu(self.conv1(concat_input))
        h = self.bn2(F.relu(self.conv2(h)))
        h = self.bn3(F.relu(self.conv3(h)))
        h = self.bn4(F.relu(self.conv4(h)))
        h = self.bn5(F.relu(self.conv5(h)))
        
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
        h = self.debn3(F.relu(self.deconv3(h)))
        h = self.debn4(F.relu(self.deconv4(h)))
        
        out = torch.sigmoid(self.deconv5(h))
        
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
        rec_mu_sigma_loss = 0
        if self.prob_decoder:
            rec_mu_sigma_loss = self.gaussian_nll(rec_mu, rec_sigma, x).sum()
        
        return rec_comps, rec, rec_mu_sigma_loss, kl


'''



class CNN_sigmaVAE(nn.Module):

    def __init__(self,latent_dim=10, window_size=32, num_feats=38, flow_type=None, use_probabilistic_decoder=False):
        super(CNN_sigmaVAE, self).__init__()
    
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.prob_decoder = use_probabilistic_decoder
        self.num_feats = num_feats
        self.flow_type = flow_type
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=5, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(4)

        #self.fc41 = nn.Linear(4*20*26, self.latent_dim)
        #self.fc42 = nn.Linear(4*20*26, self.latent_dim)
        self.fc41 = nn.Linear(4*20*(num_feats-12), self.latent_dim)
        self.fc42 = nn.Linear(4*20*(num_feats-12), self.latent_dim)

        #self.defc1 = nn.Linear(self.latent_dim, 4*20*26)
        self.defc1 = nn.Linear(self.latent_dim, 4*20*(num_feats-12))
        
        self.deconv1 = nn.ConvTranspose2d(in_channels=4, out_channels=16, kernel_size=5, stride=1, padding=0, output_padding=0)
        self.debn1 = nn.BatchNorm2d(16)
        self.deconv2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=5, stride=1, padding=0, output_padding=0)
        self.debn2 = nn.BatchNorm2d(8)
        self.deconv3 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=5, stride=1, padding=0, output_padding=0)

        self.log_sigma = 0
        self.log_sigma = torch.nn.Parameter(torch.full((1,), 0.0)[0], requires_grad=True)
        
        self.decoder_fc41 = nn.Linear(self.window_size * self.num_feats, self.window_size * self.num_feats)
        self.decoder_fc42 = nn.Linear(self.window_size * self.num_feats, self.window_size * self.num_feats)

        if self.flow_type=='RealNVP':
            self.flow = RealNVP(n_blocks=1, input_size=self.latent_dim, hidden_size=50, n_hidden=1)
        
        elif self.flow_type=='MAF':
            self.flow = MAF(n_blocks=1, input_size=self.latent_dim, hidden_size=10, n_hidden=1)
        
        elif self.flow_type =='Planar':
            self.block_planar = [PlanarFlow]
            self.flow = NormalizingFlow(dim=self.latent_dim, blocks=self.block_planar, flow_length=16, density=distrib.MultivariateNormal(torch.zeros(self.latent_dim), torch.eye(self.latent_dim)))

        elif self.flow_type =='BNAF':
            num_flows = 1
            num_layers = 2
            n_dims = 10
            hidden_dim = 10
            residual = None

            flows = []
            for f in range(num_flows):
                layers = []
                for _ in range(num_layers - 1):
                    layers.append(MaskedWeight(n_dims * hidden_dim,
                                            n_dims * hidden_dim, dim=n_dims))
                    layers.append(Tanh())

                flows.append(
                    BNAF(*([MaskedWeight(n_dims, n_dims * hidden_dim, dim=n_dims), Tanh()] + \
                        layers + \
                        [MaskedWeight(n_dims * hidden_dim, n_dims, dim=n_dims)]),\
                        res=residual if f < num_flows - 1 else None
                    )
                )

                if f < num_flows - 1:
                    flows.append(Permutation(n_dims, 'flip'))

            self.flow = Sequential(*flows).to(device)

        
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
            out = out.view(out.size(0), out.size(1), -1)
            rec_mu = self.decoder_fc41(out).tanh()
            rec_sigma = self.decoder_fc42(out).tanh()

            out = out.view(out.size(0), 1, self.window_size, self.num_feats)
            rec_mu = rec_mu.view(rec_mu.size(0), 1, self.window_size, self.num_feats)
            rec_sigma = rec_sigma.view(rec_sigma.size(0), 1, self.window_size, self.num_feats)


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
        
        if self.flow_type == 'BNAF':
            zk, loss = self.flow(z0)
        else:
            zk, loss = self.flow.log_prob(z0, None)

        loss = -loss.mean(0)

        return zk, loss


    def forward(self, x):

        z_params = self.encoder(x)
        mu, log_var = z_params

        if self.flow_type == None:
            z = self.sampling(mu, log_var)
            output, rec_mu, rec_sigma = self.decoder(z)
            kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        else:
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

    #x in input here,
    def loss_function(self, recon_x, x, rec_mu, rec_sigma, kl):
        if self.prob_decoder:
            rec_comps, rec = self.reconstruction_loss(rec_mu, x)
        else:
            rec_comps, rec = self.reconstruction_loss(recon_x, x)

        rec_mu_sigma_loss = 0
        if self.prob_decoder:
            rec_mu_sigma_loss = self.gaussian_nll(rec_mu, rec_sigma, x).sum()
        
        return rec_comps, rec, rec_mu_sigma_loss, kl