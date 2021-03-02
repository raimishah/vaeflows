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


# 5 layer -- 100 window size

class TCN_VAE(nn.Module):

    def __init__(self,conditional=False, latent_dim=10, window_size=32, cond_window_size=32, jump_size=32, num_feats=38, kernel_size=5, num_levels=2, convs_per_level = 2, flow_type=None, use_probabilistic_decoder=False):
        super(TCN_VAE, self).__init__()
    
        self.conditional = conditional
        self.window_size = window_size
        self.cond_window_size = cond_window_size
        self.jump_size = jump_size
        self.latent_dim = latent_dim
        self.prob_decoder = use_probabilistic_decoder
        self.num_feats = num_feats
        self.flow_type = flow_type
        self.kernel_size = kernel_size
        self.num_levels = num_levels
        self.convs_per_level = convs_per_level
        self.before_pool_dims = []
        self.last_encoder_conv_dims = (-1,-1)

        self.conditional_entry_place_dimensions = None

        self.channels = [int(2**i) for i in range(0,num_levels*convs_per_level+1)]
        channel_idx = 1

        #ENCODER
        dilation = 2
        layers = []
        cur_H = window_size
        cur_W = num_feats
        for i in range(1, self.num_levels+1):

            if i == 2:
                self.conditional_entry_place_dimensions = (self.channels[channel_idx-1], int(cur_H), int(cur_W))

            for j in range(convs_per_level):
                cur_dilation = (dilation**(i-1), 1)
                constant_padding = (0,0,dilation**(i-1)*(self.kernel_size-1),0)

                pad = nn.ConstantPad2d(padding = constant_padding, value=0.0)
                conv = nn.Conv2d(in_channels=self.channels[channel_idx-1], out_channels=self.channels[channel_idx], kernel_size=(self.kernel_size, self.kernel_size), dilation=cur_dilation)
                batchnorm = nn.BatchNorm2d(num_features=self.channels[channel_idx])
                relu = nn.ReLU6()
                dropout = nn.Dropout(p=.2)
                
                layers.append(pad)
                layers.append(conv)
                layers.append(batchnorm)
                layers.append(relu)
                layers.append(dropout)
                channel_idx += 1

                cur_H += constant_padding[2]                 
                cur_W += 0

                #insert conditional data here, so dimensions match and can preprocess cond data in same way
                if self.conditional and i == 2 and j == 0:
                    cur_H += 1
                
                cur_H = np.floor((cur_H + 2*0 - cur_dilation[0] * (self.kernel_size-1) - 1)/1 + 1 )
                cur_W = np.floor((cur_W - 2*0 - cur_dilation[1] * (self.kernel_size-1) - 1)/1 + 1)

                
            if i <= num_levels - 1: # and i % 2 == 0:
                layers.append(nn.MaxPool2d(kernel_size=(2,2), ceil_mode=True))
                self.before_pool_dims.append((int(cur_H), int(cur_W)))
                cur_H = np.ceil(cur_H / 2)
                cur_W = np.ceil(cur_W / 2)
            
            
        self.encoder_net = nn.Sequential(*layers)

        self.last_encoder_conv_dims = (int(cur_H), int(cur_W))

        lin_layer = cur_H*cur_W

        self.fc41 = nn.Linear(int(self.channels[-1]*lin_layer), self.latent_dim)
        self.fc42 = nn.Linear(int(self.channels[-1]*lin_layer), self.latent_dim)

        self.defc1 = nn.Linear(self.latent_dim, int(self.channels[-1]*lin_layer))
        



        #DECODER
        channel_idx = len(self.channels)-1

        layers = []
        for i in range(self.num_levels, 0, -1):

            if i <= num_levels - 1 and len(self.before_pool_dims) >= 1:
                upsample = nn.Upsample(size=(self.before_pool_dims[-1]), mode='nearest')
                self.before_pool_dims.pop(-1)
                layers.append(upsample)

            for j in range(convs_per_level):

                constant_padding = (0,0,dilation**(num_levels-i)*(self.kernel_size-1),0)
                cur_dilation = (dilation**(num_levels-i), 1)
                conv_padding = (constant_padding[2], 0)

                pad = nn.ConstantPad2d(padding = constant_padding, value=0.0)

                #if i == 1:
                #   conv = nn.ConvTranspose2d(in_channels=self.channels[channel_idx], out_channels=self.channels[channel_idx-1], kernel_size=(self.kernel_size,self.kernel_size), padding=(conv_padding[0]+int((self.kernel_size-3)/2),0), dilation=cur_dilation)
                #else:
                conv = nn.ConvTranspose2d(in_channels=self.channels[channel_idx], out_channels=self.channels[channel_idx-1], kernel_size=(self.kernel_size,self.kernel_size), padding=conv_padding, dilation=cur_dilation)
                
                batchnorm = nn.BatchNorm2d(num_features=self.channels[channel_idx-1])
                relu = nn.ReLU6()
                dropout = nn.Dropout(p=.2)


                layers.append(pad)
                layers.append(conv)
                layers.append(batchnorm)
                if i == 1 and j == convs_per_level - 1: # last layer sigmoid for 0-1 output
                    layers.append(nn.Sigmoid())
                else:
                    layers.append(relu)
                    layers.append(dropout)
                channel_idx -= 1
        
        self.decoder_net = nn.Sequential(*layers)


        if self.conditional:
            #conditional decreasing network
            layers = []
            
            convs_per = 1
            if convs_per == 2:
                self.cond_channels = [1, self.conditional_entry_place_dimensions[0] // 2, self.conditional_entry_place_dimensions[0]]
            elif convs_per==1:
                self.cond_channels = [1,  self.conditional_entry_place_dimensions[0]]

            cond_channel_idx = 1
            cur_H = self.cond_window_size
            cur_W = self.num_feats

            for i in range(1):
                for j in range(convs_per):
                    conv = nn.Conv2d(in_channels=self.cond_channels[cond_channel_idx-1], out_channels=self.cond_channels[cond_channel_idx], kernel_size=(self.kernel_size, self.kernel_size))
                    bn = nn.BatchNorm2d(num_features=self.cond_channels[cond_channel_idx])
                    relu = nn.ReLU() #regular ReLU should be fine here

                    layers.append(conv)
                    layers.append(bn)
                    layers.append(relu)
                    cond_channel_idx+=1

                    cur_H = np.floor((cur_H + 2*0 - 1 * (self.kernel_size-1) - 1)/1 + 1 )
                    cur_W = np.floor((cur_W - 2*0 - 1 * (self.kernel_size-1) - 1)/1 + 1)

                    #if i != 2-1:
                    layers.append(nn.MaxPool2d(kernel_size=(2,2), ceil_mode=True))
                    cur_H = np.ceil(cur_H / 2)
                    cur_W = np.ceil(cur_W / 2)

            layers.append(nn.Flatten())
            layers.append(nn.Linear(int(self.cond_channels[-1]*cur_H*cur_W), int(self.conditional_entry_place_dimensions[0] * 1 * self.conditional_entry_place_dimensions[2])))
            #layers.append(nn.Sigmoid()) #sigmoid/relu shouldnt really matter but leave it for now

            self.conditional_net = nn.Sequential(*layers)
        







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
            n_dims = self.latent_dim
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

        
        
    def encoder(self, x, c):
        if self.conditional:
            c = self.conditional_net(c)
            c = c.view(c.shape[0], self.conditional_entry_place_dimensions[0], 1, int(self.conditional_entry_place_dimensions[2]))
        
        
        #h = self.encoder_net(x)
        h = x
        for layer in self.encoder_net:
            h = layer(h)
            if self.conditional:
                if h.shape[1:] == self.conditional_entry_place_dimensions:
                    h = torch.cat([h, c], 2)

        h = h.view(h.size(0), -1)
        
        mu, var = self.fc41(h), self.fc42(h)
                
        return self.fc41(h), self.fc42(h)

    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu) # return z sample
    
    def decoder(self, z, c):

        z = self.defc1(z)
        z = z.view(z.size(0), self.channels[-1], self.last_encoder_conv_dims[0], self.last_encoder_conv_dims[1])

        if self.conditional:
            c = self.conditional_net(c)
            c = c.view(c.shape[0], self.conditional_entry_place_dimensions[0], 1, int(self.conditional_entry_place_dimensions[2]))

        h = z

        for layer in self.decoder_net:
            h = layer(h)
            #print('decoder shape : {}'.format(h.shape))
            if self.conditional:
                #print(h.shape, self.conditional_entry_place_dimensions, '\n')
                if h.shape[1] == self.conditional_entry_place_dimensions[0] and h.shape[2]-1 == self.conditional_entry_place_dimensions[1] and h.shape[3] == self.conditional_entry_place_dimensions[2]:
                    h = torch.cat([h, c], 2)

        out = h


        #out = self.decoder_net(concat_input)




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


        
    def forward(self, x, c):


        #print(x.shape)
        z_params = self.encoder(x, c)
        mu, log_var = z_params

        #print(mu.shape)
        if self.flow_type == None:
            z = self.sampling(mu, log_var)
            output, rec_mu, rec_sigma = self.decoder(z, c)
            kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        else:
            if self.flow_type =='Planar':
                z_k, kl = self.latent_planar(x, z_params)
            else:
                z_k, kl = self.latent_not_planar(x, z_params)
            
            output, rec_mu, rec_sigma = self.decoder(z_k, c)
    
        #print(output.shape)

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

