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
import naf



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
class CNN_sigmacVAE(nn.Module):

    def __init__(self,latent_dim=8, window_size=20, cond_window_size=10, num_feats=38, flow_type=None, use_probabilistic_decoder=False):
        super(CNN_sigmacVAE, self).__init__()
        
        self.window_size=window_size
        self.cond_window_size=cond_window_size
        self.latent_dim = latent_dim
        self.num_feats = num_feats
        self.flow_type = flow_type
        self.prob_decoder = use_probabilistic_decoder
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(3,3), stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(3,3), stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=(3,3), stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(4)

        self.pool2x2 = nn.MaxPool2d((2,2), return_indices=True)
        self.unpool2x2 = nn.MaxUnpool2d((2,2))
        self.pool2x2_idxs = []

        lin_layer= 54*11
        self.fc41 = nn.Linear(4*lin_layer, self.latent_dim)
        self.fc42 = nn.Linear(4*lin_layer, self.latent_dim)

        self.defc1 = nn.Linear(self.latent_dim, 4*lin_layer)
        
        self.deconv1 = nn.ConvTranspose2d(in_channels=4, out_channels=16, kernel_size=(3,3), stride=1, padding=0, output_padding=0)
        self.debn1 = nn.BatchNorm2d(16)
        self.deconv2 = nn.ConvTranspose2d(in_channels=16, out_channels=64, kernel_size=(3,3), stride=1, padding=0, output_padding=0)
        self.debn2 = nn.BatchNorm2d(64)
        self.deconv2_extra = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=0, output_padding=0)
        self.debn2_extra = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3,3), stride=1, padding=0, output_padding=0)
        self.debn3 = nn.BatchNorm2d(32)
        self.deconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(3,3), stride=1, padding=0, output_padding=0)
        self.debn4 = nn.BatchNorm2d(8)
        self.deconv5 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(3,3), stride=1, padding=0, output_padding=0)        
        
        
        self.log_sigma = 0
        self.log_sigma = torch.nn.Parameter(torch.full((1,), 0.0)[0], requires_grad=True)
        
        self.decoder_fc41 = nn.Linear(self.window_size * self.num_feats, self.window_size * self.num_feats)
        self.decoder_fc42 = nn.Linear(self.window_size * self.num_feats, self.window_size * self.num_feats)
        

        if self.flow_type=='RealNVP':
            #self.flow = RealNVP(n_blocks=1, input_size=self.latent_dim, hidden_size=50, n_hidden=1)
            self.flow = RealNVP(n_blocks=1, input_size=self.latent_dim, cond_label_size = self.cond_window_size * self.num_feats, hidden_size=50, n_hidden=1)
        
        elif self.flow_type=='MAF':
            #self.flow = MAF(n_blocks=1, input_size=self.latent_dim, hidden_size=10, n_hidden=1)
            self.flow = MAF(n_blocks=1, input_size=self.latent_dim, cond_label_size = self.cond_window_size * self.num_feats, hidden_size=10, n_hidden=1)
        
        elif self.flow_type =='Planar':
            self.block_planar = [PlanarFlow]
            self.flow = NormalizingFlow(dim=self.latent_dim, blocks=self.block_planar, flow_length=16, density=distrib.MultivariateNormal(torch.zeros(self.latent_dim), torch.eye(self.latent_dim)))
        
        elif self.flow_type=='DSF':
            #self.lgd = torch.autograd.Variable(torch.from_numpy(np.random.rand(self.window_size).astype('float32')))    
            hidden_dim = 100
            self.flow = naf.IAF_DSF(dim=latent_dim, hid_dim=hidden_dim, context_dim=self.cond_window_size*self.num_feats, num_layers=3)

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

        
    def encoder(self, x, c):    
        concat_input = torch.cat([x, c], 2)
        

        h = F.relu(self.conv1(concat_input))
        print('after conv1: {}'.format(h.shape))
        h = self.bn2(F.relu(self.conv2(h)))
        print('after conv2: {}'.format(h.shape))
        h, pool_idxs = self.pool2x2(h)
        self.pool2x2_idxs.append(pool_idxs)
        print('shape after pooling {}'.format(h.shape))

        h= self.bn3(F.relu(self.conv3(h)))
        print('after conv3: {}'.format(h.shape))
        #end here after 3 layers

        h = self.bn4(F.relu(self.conv4(h)))
        print('after conv4: {}'.format(h.shape))
        h = self.bn5(F.relu(self.conv5(h)))
        print('after conv5: {}'.format(h.shape))

        self.saved_dim = [h.size(1), h.size(2), h.size(3)]
        

        h = h.view(h.size(0), -1)
        #h = h.view(h.size(0), h.size(2) * h.size(3))
        
        mu, var = self.fc41(h), self.fc42(h)
                
        return self.fc41(h), self.fc42(h)

    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu) # return z sample
    
    def decoder(self, z, c):
        print(self.pool2x2_idxs[0].shape)

        c = c.view(c.size(0), -1)
        print(z.shape)
        print(c.shape)
        concat_input = torch.cat([z,c],1)
        print(concat_input.shape)
        
        print('^^^^ This shape[1] divided by most recent out channels then reshape')
        
        concat_input = concat_input.view(concat_input.size(0), self.saved_dim[0], 52, 9)
        
        
        h = self.debn1(F.relu(self.deconv1(concat_input)))
        print('after deconv1: {}'.format(h.shape))
        h = self.debn2(F.relu(self.deconv2(h)))
        print('after deconv2: {}'.format(h.shape))
        h = self.debn2_extra(F.relu(self.deconv2_extra(h)))
        print('after deconv2: {}'.format(h.shape))
        #out = torch.sigmoid(self.deconv3(h))
        #end here after 3 layers

        
        h = self.debn3(F.relu(self.deconv3(h)))
        print('after deconv3: {}'.format(h.shape))
        h = self.unpool2x2(h, self.pool2x2_idxs[-1])
        self.pool2x2_idxs.pop()
        print('after unpool: {}'.format(h.shape))
        
        h = self.debn4(F.relu(self.deconv4(h)))
        
        #h = self.unpool2x2(self.debn4(F.relu(self.deconv4(h))), self.pool2x2_idxs[-1])
        #self.pool2x2_idxs.pop()
        print('after deconv4: {}'.format(h.shape))
        out = torch.sigmoid(self.deconv5(h))
        print('after last deconv: {}'.format(out.shape))


        if self.prob_decoder:
            out = out.view(out.size(0), out.size(1), -1)
            print(out.shape)
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

    
    def latent_not_planar(self, x, z_params, c):
        n_batch = x.size(0)
                
        # Retrieve set of parameters
        #mu, sigma = z_params
        mu, log_var = z_params
        sigma = torch.sqrt(log_var.exp())
        
        # Obtain our first set of latent points
        z0 = self.sampling(mu, log_var)
        
        #zk, loss = self.flow.log_prob(z0, None)
        c = c.view(c.size(0), -1)
        
        if self.flow_type == 'BNAF':
            zk, loss = self.flow(z0)
        elif self.flow_type=='DSF':
            #print(z0.shape, self.lgd.shape, c.shape)
            self.lgd = torch.autograd.Variable(torch.from_numpy(np.random.rand(z0.shape[0]).astype('float32')))    
            zk, loss, _ = self.flow.forward((z0, self.lgd, c))
            #print(self.inp.shape, self.lgd.shape, self.con.shape)
            #zk, loss, _ = self.flow.forward((self.inp, self.lgd, self.con))
            #zk = z0

        else:
            zk, loss = self.flow.log_prob(z0, c)
        
        loss = -loss.mean(0)

        return zk, loss


        
    def forward(self, x, c):

        z_params = self.encoder(x, c)
        mu, log_var = z_params

        if self.flow_type == None:
            z = self.sampling(mu, log_var)
            output, rec_mu, rec_sigma = self.decoder(z, c)
            kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        else:
            if self.flow_type =='Planar':
                z_k, kl = self.latent_planar(x, z_params)
            else:
                z_k, kl = self.latent_not_planar(x, z_params, c)
            
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



    def generate(self, c):
        #if torch.cuda.is_available():
        c = c.to(device)

        mu = torch.zeros(c.shape[0], self.latent_dim).to(device)
        log_var = torch.zeros(c.shape[0], self.latent_dim).to(device)
        
        z_params = (mu, log_var)

        if self.flow_type == None:
            z = self.sampling(mu, log_var)
            output, rec_mu, rec_sigma = self.decoder(z, c)
            kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        else:
            if self.flow_type =='Planar':
                z_k, kl = self.latent_planar(c, z_params)
            else:
                z_k, kl = self.latent_not_planar(c, z_params, c)
            
            output, rec_mu, rec_sigma = self.decoder(z_k, c)
    
        return output, rec_mu, rec_sigma, kl


'''


class CNN_sigmacVAE(nn.Module):

    def __init__(self,latent_dim=8, window_size=20, cond_window_size=10, num_feats=38, flow_type=None, use_probabilistic_decoder=False):
        super(CNN_sigmacVAE, self).__init__()
        
        self.window_size=window_size
        self.cond_window_size=cond_window_size
        self.latent_dim = latent_dim
        self.num_feats = num_feats
        self.flow_type = flow_type
        self.prob_decoder = use_probabilistic_decoder
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=5, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(4)

        self.fc41 = nn.Linear(4*33*26, self.latent_dim)
        self.fc42 = nn.Linear(4*33*26, self.latent_dim)

        self.deconv1 = nn.ConvTranspose2d(in_channels=4, out_channels=16, kernel_size=(9,9), stride=1, padding=0, output_padding=0)
        self.debn1 = nn.BatchNorm2d(16)
        self.deconv2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(9,9), stride=1, padding=0, output_padding=0)
        self.debn2 = nn.BatchNorm2d(8)
        self.deconv3 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(8,9), stride=1, padding=0, output_padding=0)

        self.log_sigma = 0
        self.log_sigma = torch.nn.Parameter(torch.full((1,), 0.0)[0], requires_grad=True)
        
        self.decoder_fc41 = nn.Linear(self.window_size * self.num_feats, self.window_size * self.num_feats)
        self.decoder_fc42 = nn.Linear(self.window_size * self.num_feats, self.window_size * self.num_feats)
        
        
        self.latent_cond_layer = nn.Linear(self.cond_window_size * self.num_feats, self.latent_dim)
        
        

        if self.flow_type=='RealNVP':
            #self.flow = RealNVP(n_blocks=1, input_size=self.latent_dim, hidden_size=50, n_hidden=1)
            self.flow = RealNVP(n_blocks=1, input_size=self.latent_dim, cond_label_size = self.cond_window_size * self.num_feats, hidden_size=50, n_hidden=1)
        
        elif self.flow_type=='MAF':
            #self.flow = MAF(n_blocks=1, input_size=self.latent_dim, hidden_size=10, n_hidden=1)
            self.flow = MAF(n_blocks=1, input_size=self.latent_dim, cond_label_size = self.cond_window_size * self.num_feats, hidden_size=10, n_hidden=1)
        
        elif self.flow_type =='Planar':
            self.block_planar = [PlanarFlow]
            self.flow = NormalizingFlow(dim=self.latent_dim, blocks=self.block_planar, flow_length=16, density=distrib.MultivariateNormal(torch.zeros(self.latent_dim), torch.eye(self.latent_dim)))

        elif self.flow_type=='DSF':
            #self.lgd = torch.autograd.Variable(torch.from_numpy(np.random.rand(self.window_size).astype('float32')))    
            hidden_dim = 100
            self.flow = naf.IAF_DSF(dim=latent_dim, hid_dim=hidden_dim, context_dim=self.cond_window_size*self.num_feats, num_layers=3)
        
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

        
        
    def encoder(self, x, c):
        
        concat_input = torch.cat([x, c], 2)
        
        h = F.relu(self.conv1(concat_input))
        h = self.bn2(F.relu(self.conv2(h)))
        h = self.bn3(F.relu(self.conv3(h)))
        
        self.saved_dim = [h.size(1), h.size(2), h.size(3)]
        

        h = h.view(h.size(0), -1)
        #h = h.view(h.size(0), h.size(2) * h.size(3))
        
        mu, var = self.fc41(h), self.fc42(h)
                
        return self.fc41(h), self.fc42(h)

    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu) # return z sample
    
    def decoder(self, z, c):

        c = c.view(c.size(0), -1)
        concat_input = torch.cat([z,c],1)
        
        concat_input = concat_input.view(concat_input.size(0), self.saved_dim[0], 9, 14)
        
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

    
    def latent_not_planar(self, x, z_params, c):
        n_batch = x.size(0)
                
        # Retrieve set of parameters
        #mu, sigma = z_params
        mu, log_var = z_params
        sigma = torch.sqrt(log_var.exp())
        
        # Obtain our first set of latent points
        z0 = self.sampling(mu, log_var)
        
        #zk, loss = self.flow.log_prob(z0, None)
        c = c.view(c.size(0), -1)
        
        if self.flow_type == 'BNAF':
            zk, loss = self.flow(z0)
        elif self.flow_type=='DSF':
            #print(z0.shape, self.lgd.shape, c.shape)
            self.lgd = torch.autograd.Variable(torch.from_numpy(np.random.rand(z0.shape[0]).astype('float32')))    
            zk, loss, _ = self.flow.forward((z0, self.lgd, c))
            #print(self.inp.shape, self.lgd.shape, self.con.shape)
            #zk, loss, _ = self.flow.forward((self.inp, self.lgd, self.con))
            #zk = z0

        else:
            zk, loss = self.flow.log_prob(z0, c)
        
        loss = -loss.mean(0)

        return zk, loss


        
    def forward(self, x, c):

        z_params = self.encoder(x, c)
        mu, log_var = z_params

        if self.flow_type == None:
            z = self.sampling(mu, log_var)
            output, rec_mu, rec_sigma = self.decoder(z, c)
            kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        else:
            if self.flow_type =='Planar':
                z_k, kl = self.latent_planar(x, z_params)
            else:
                z_k, kl = self.latent_not_planar(x, z_params, c)
            
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



    def generate(self, c):
        #if torch.cuda.is_available():
        c = c.to(device)

        mu = torch.zeros(c.shape[0], self.latent_dim).to(device)
        log_var = torch.zeros(c.shape[0], self.latent_dim).to(device)
        
        z_params = (mu, log_var)

        if self.flow_type == None:
            z = self.sampling(mu, log_var)
            output, rec_mu, rec_sigma = self.decoder(z, c)
            kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        else:
            if self.flow_type =='Planar':
                z_k, kl = self.latent_planar(x, z_params)
            else:
                z_k, kl = self.latent_not_planar(x, z_params, c)
            
            output, rec_mu, rec_sigma = self.decoder(z_k, c)
    
        return output, rec_mu, rec_sigma, kl
