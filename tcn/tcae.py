import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F


class Encoder1D(nn.Module):
    def __init__(self, window_size=32, num_feats=1, num_levels=3, kernel_size=3):
        super(Encoder1D, self).__init__()

        #pad
        self.kernel_size=kernel_size
        self.num_levels = num_levels

        dilation = 2
        layers = []
        for i in range(1, self.num_levels+1):
            pad = nn.ConstantPad1d(padding = (dilation**i, 0), value=0.0)
            conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=self.kernel_size, dilation=dilation**(i-1))
            relu = nn.ReLU()
            maxpool = nn.MaxPool1d(kernel_size=2, ceil_mode=True)
            layers.append(pad)
            layers.append(conv)
            layers.append(relu)
            if i <= num_levels - 2: # and i % 2 == 0:
                layers.append(maxpool)
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        print('\n\n')

        print(self.net)

        print('\n\n')
        print('input x : {}'.format(x.shape))
        pad_count = 0
        conv_count = 0
        pool_count = 0
        for layer_idx, layer in enumerate(self.net):
            x = layer(x)
            
            if isinstance(layer, nn.ConstantPad1d):
                print('after pad{} {}'.format(pad_count, x.shape))
                pad_count+=1
            if isinstance(layer, nn.Conv1d):
                print('after conv{} {}'.format(conv_count,x.shape))
                conv_count+=1
            if isinstance(layer, nn.MaxPool1d):
                print('after maxpool{} {}'.format(pool_count, x.shape))
                pool_count+=1
        
        print('\n\n')
        return x



class Decoder1D(nn.Module):
    def __init__(self, window_size=32, num_feats=1, num_levels=3, kernel_size=3):
        super(Decoder1D, self).__init__()

        #pad
        self.kernel_size=kernel_size
        self.num_levels=num_levels

        dilation = 2
        layers = []
        for i in range(self.num_levels, 0, -1):
            upsample = nn.Upsample(scale_factor=2)

            #pad = nn.ConstantPad1d(padding = (dilation**(num_levels-i), 0), value=0.0) if i < -1 else nn.ConstantPad1d(padding = (0, 0), value=0.0)
            #conv = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=self.kernel_size, padding=dilation**(num_levels-i), dilation=dilation**(num_levels-i))
            #this or that^^
            #this below i think preserves causality moreso?
            pad = nn.ConstantPad1d(padding = (dilation**(num_levels-i)*2, 0), value=0.0)
            conv = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=self.kernel_size, padding=dilation**(num_levels-i)*2, dilation=dilation**(num_levels-i))
            relu = nn.ReLU()
            upsample = nn.Upsample(scale_factor=2)
            if i <= num_levels-2:
                layers.append(upsample)
            layers.append(pad)
            layers.append(conv)
            if i >= 2:
                layers.append(relu)
            else:
                layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        print('\n\n')

        print(self.net)

        print('\n\n')
        print('input x : {}'.format(x.shape))
        pad_count = 0
        conv_count = 0
        upsample_count = 0
        for layer_idx, layer in enumerate(self.net):
            x = layer(x)
            
            if isinstance(layer, nn.ConstantPad1d):
                print('after pad{} {}'.format(pad_count, x.shape))
                pad_count+=1
            if isinstance(layer, nn.ConvTranspose1d):
                print('after upconv{} {}'.format(conv_count,x.shape))
                conv_count+=1
            if isinstance(layer, nn.Upsample):
                print('after upsample{} {}'.format(upsample_count, x.shape))
                upsample_count+=1
        
        print('\n\n')
        return x




#x_1d = torch.ones((256, 1, 32))
#enc_model = Encoder1D(num_levels=4)
#encoder_out = enc_model(x_1d)
#dec_model = Decoder1D(num_levels=4)
#decoder_out = dec_model(encoder_out)




class TCAEEncoder2D(nn.Module):
    def __init__(self, window_size=32, num_feats=1, num_levels=3, kernel_size=3):
        super(TCAEEncoder2D, self).__init__()

        #pad
        self.kernel_size=kernel_size
        self.num_levels = num_levels

        self.before_pool_dims = []

        dilation = 2
        layers = []
        cur_H = window_size
        cur_W = num_feats
        for i in range(1, self.num_levels+1):

            cur_dilation = (dilation**(i-1), 1)
            #print(cur_dilation)
            constant_padding = (0,0,dilation**i,0)

            pad = nn.ConstantPad2d(padding = constant_padding, value=0.0)
            conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(self.kernel_size, self.kernel_size), dilation=cur_dilation)
            relu = nn.ReLU()
            maxpool = nn.MaxPool2d(kernel_size=(2,2), ceil_mode=True)
            layers.append(pad)
            layers.append(conv)
            layers.append(relu)
            if i <= num_levels - 2: # and i % 2 == 0:
                layers.append(maxpool)

            cur_H += constant_padding[2]
            cur_W += 0
            print(cur_H, cur_W)
            cur_H = np.floor((cur_H + 2*0 - cur_dilation[0] * (self.kernel_size-1) - 1)/ 1 + 1 )
            cur_W = np.floor((cur_W - 2*0 - cur_dilation[1] * (self.kernel_size-1) - 1)/1 + 1)
            print(cur_H, cur_W)
            if i <= num_levels - 2:
                self.before_pool_dims.append((int(cur_H), int(cur_W)))
                cur_H = np.ceil(cur_H / 2)
                cur_W = np.ceil(cur_W / 2)

            print(cur_H, cur_W)
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        print('\n\n')

        print(self.net)

        print('\n\n')
        print('input x : {}'.format(x.shape))
        pad_count = 0
        conv_count = 0
        pool_count = 0
        for layer_idx, layer in enumerate(self.net):
            
            x = layer(x)
            #print(layer)
            if isinstance(layer, nn.ConstantPad2d):
                print('after pad{} {}'.format(pad_count, x.shape))
                pad_count+=1
            if isinstance(layer, nn.ReLU) or isinstance(layer, nn.Conv2d):
                print('after conv{} {}'.format(conv_count,x.shape))
                conv_count+=1
            if isinstance(layer, nn.MaxPool2d):
                print('after maxpool{} {}'.format(pool_count, x.shape))
                pool_count+=1
        
        print('\n\n')
        return x



class TCAEDecoder2D(nn.Module):
    def __init__(self, before_pool_dims, window_size=32, num_feats=1, num_levels=3, kernel_size=3):
        super(TCAEDecoder2D, self).__init__()

        #pad
        self.kernel_size=kernel_size
        self.num_levels=num_levels

        dilation = 2
        layers = []
        for i in range(self.num_levels, 0, -1):
            upsample = nn.Upsample(scale_factor=2)

            #pad = nn.ConstantPad2d(padding = (dilation**(num_levels-i), 0), value=0.0) if i < -1 else nn.ConstantPad1d(padding = (0, 0), value=0.0)
            #conv = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(self.kernel_size,self.kernel_size), padding=dilation**(num_levels-i), dilation=dilation**(num_levels-i))
            #this or that^^
            #this below i think preserves causality moreso?
            pad = nn.ConstantPad2d(padding = (0, 0, dilation**(num_levels-i)*2, 0), value=0.0)
            conv = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(self.kernel_size,self.kernel_size), padding=(dilation**(num_levels-i)*2, 0), dilation=(dilation**(num_levels-i), 1))
            relu = nn.ReLU()
            if i <= len(before_pool_dims) >= 1:
                upsample = nn.Upsample(size=(before_pool_dims[-1]), mode='nearest')
                before_pool_dims.pop(-1)
                layers.append(upsample)
            layers.append(pad)
            layers.append(conv)
            if i >= 2:
                layers.append(relu)
            else:
                layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        print('\n\n')

        print(self.net)

        print('\n\n')
        print('input x : {}'.format(x.shape))
        pad_count = 0
        conv_count = 0
        upsample_count = 0
        for layer_idx, layer in enumerate(self.net):            
            x = layer(x)

            if isinstance(layer, nn.ConstantPad2d):
                print('after pad{} {}'.format(pad_count, x.shape))
                pad_count+=1
            if isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.ReLU):
                print('after upconv{} {}'.format(conv_count,x.shape))
                conv_count+=1
            if isinstance(layer, nn.Upsample):
                print('after upsample{} {}'.format(upsample_count, x.shape))
                upsample_count+=1
        
        print('\n\n')
        return x

print('-------------------------------------2D------------------------------------')


window_size=100
num_feats=55
kernel_size=3
num_levels=4
x_2d = torch.ones((256, 1, window_size, num_feats))

#enc_model = TCAEEncoder2D(num_levels=num_levels, window_size=window_size, num_feats=num_feats, kernel_size=3)
#encoder_out = enc_model(x_2d)
#before_pool_dims = enc_model.before_pool_dims
#dec_model = TCAEDecoder2D(before_pool_dims, num_levels=num_levels)
#decoder_out = dec_model(encoder_out)


class TCAE(nn.Module):
    def __init__(self, num_levels=3, window_size=window_size, num_feats=num_feats, kernel_size=3):
        super(TCAE, self).__init__()

        self.encoder = TCAEEncoder2D(num_levels=num_levels, window_size=window_size, num_feats=num_feats, kernel_size=3)
        before_pool_dims = self.encoder.before_pool_dims
        self.decoder = TCAEDecoder2D(before_pool_dims, num_levels=num_levels)


    def forward(self, x):
        enc_out = self.encoder(x)
        dec_out = self.decoder(enc_out)
        return dec_out


ae = TCAE(num_levels=num_levels, window_size=window_size, num_feats=num_feats, kernel_size=3)
ae(x_2d)

#CVAE archs will need to be adjusted...

