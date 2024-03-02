#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 15:28:22 2023

@author: dan
"""
from time  import time

import torch
from torch import nn, Tensor
import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class MLP(torch.nn.Module):
    def __init__(self, dims, actfns = [], bias = [], use_smax = False):
        nn.Module.__init__(self)
        
        # Number of layers, layers, activation functions
        self.n_layers = len(dims) - 1
        self.actfns = []
        self.layers = nn.ModuleList()
        
        self.smax  = nn.Softmax(dim=1)
        self.use_smax = use_smax

        
        # Define default activation functions
        if len(actfns) == 0:
            actfns = ['identity' for i in dims[1:]]
            
        # Default baises
        if len(bias) == 0:
            bias = [True for i in dims[1:]]
        
        # Define layers
        for i, (dim0, dim1) in enumerate( zip(dims, dims[1:]) ):
            self.layers.append( nn.Linear(dim0, dim1, bias = bias[i]) )
            self.actfns.append( get_actfn( actfns[i]) )
            
    # Standard forward propagation, nothing to see here.
    def forward(self, x):
        
        # Run input through layers
        for lnum in range(self.n_layers):
            x = self.actfns[lnum](self.layers[lnum](x))
        
        # Softmax?
        if self.use_smax:
            x = self.smax(x)
        
        # Return output
        return x


class AE(torch.nn.Module):
    def __init__(self, dims, actfns = [], bias = [], use_smax = False):
        nn.Module.__init__(self)
        
        # Number of layers, layers, activation functions
        self.n_layers = len(dims) - 1
        self.actfns = []
        self.wgts = nn.ParameterList()
        self.bias = nn.ParameterList()
        self.dims = dims
        
        # Use softmax?
        self.use_smax = use_smax
        if use_smax:
            self.smax  = nn.Softmax(dim=2)

        # Define default activation functions
        if len(actfns) == 0:
            actfns = ['tanh' for i in dims[1:]]
            
        # Default baises
        if len(bias) == 0:
            bias = [False for i in dims[1:]]
        
        # Define layers
        for i, (dim0, dim1) in enumerate( zip(dims, dims[1:]) ):
            # Weights and biaess
            self.wgts.append( nn.Parameter( torch.randn(dim0, dim1)/dim1 ) )
            self.bias.append( nn.Parameter( torch.randn(      dim1)/dim1 ) )
            
            # Activation functions
            self.actfns.append( get_actfn( actfns[i]) )
    
    # Standard forward propagation, nothing to see here.
    def forward(self, x):
        offset = torch.mean(x)
        std    = torch.std(x)
        x      = (x - offset)/std
    
        # Run input through encoder
        for lnum in range(self.n_layers):
            x = torch.nn.functional.linear(x, self.wgts[lnum].T )#, self.bias[lnum]))        

        # Run hidden through decoder
        for lnum in range(self.n_layers-1, -1, -1):
            x = torch.nn.functional.linear(x, self.wgts[lnum]   )#, self.bias[lnum]))            

        # Return output
        return x*std + offset


class Transformer(nn.Module):
    def __init__(self, dim_in, dim_out, dim_emb, nhead, nlayers):
        super(Transformer, self).__init__()

        # Define some input kernel parameters
        nkers   = 10
        kstride = 100
        hlens   = [dim_in  for i in range(nkers)]
        strides = [kstride for i in range(nkers)]
        kwidths = range(100, 100*(nkers+1), 100)

        # Calculate tokenizer output size
        self.dim_token_out = int(dim_in/kstride * nkers)

        # Input kernels themselves
        self.kernel_blocks = KernelBlocks(hlens = hlens, strides = strides, kwidths = kwidths)

        # Define tokenizer based on kernels
        self.tokenizer = PseudoTokenizer(self.kernel_blocks)

        # Embedding layer
        self.embed = nn.Linear(self.dim_token_out, dim_emb)

        # Transformer
        self.memory = torch.zeros((1,dim_emb))
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model = dim_emb, nhead = nhead),
            num_layers = nlayers
        )

        # Readout
        self.unembed = nn.Linear(dim_emb, dim_out)

    def forward(self, input):
        # Convert input to tokens
        tokens = self.tokenizer.tokenize(input)
        tokens = torch.concatenate(tokens, axis=1)

        # Token embeddings
        latent = self.embed(tokens)

        # Transformer
        output = self.transformer_decoder(latent, self.memory)

        # Predictions
        return self.unembed(output)



class KernelBlocks(nn.Module):
    def __init__(self, hlens = [5000], strides = [10], kwidths = [200]):
        super().__init__()

        # For each kernel: history length, stride, kernel width
        self.nkers   = len(hlens)
        self.hlens   = hlens
        self.strides = strides
        self.kwidths = kwidths

        # Define envelope for localizing each kernel
        self.envs = []
        for i in range(0, self.nkers):
            self.envs.append(norm.pdf(np.linspace(-3, 3, kwidths[i]), loc = 0, scale = 1))

        # Kernels. Conv1D requires dims (channels, channels/group, kwidth)
        self.kernels = []
        for i in range(0, self.nkers):
            self.kernels.append(torch.randn(1,1, kwidths[i]))

        # Restrict kernels to envelopes
        self.trunc_to_env()


    # Call for coercing kernels into envelopes
    def trunc_to_env(self):
        
        # For every kernel
        for i in range(self.nkers):

            # Loop over kernel dim and truncate
            for j in range(0, self.kwidths[i]):
            
                # Ugly indexing & probably inefficient.
                # Blame it on the bossa nova.
                if self.kernels[i][0,0,j] >= self.envs[i][j]:
                    self.kernels[i][0,0,j] = self.envs[i][j]
                elif self.kernels[i][0,0,j] <= -self.envs[i][j]:
                    self.kernels[i][0,0,j] = -self.envs[i][j]


#
class PseudoTokenizer(nn.Module):
    def __init__(self, kblocks):
        super().__init__()

        # Probably a better way to do this
        self.nkers   = kblocks.nkers
        self.hlens   = kblocks.hlens
        self.strides = kblocks.strides
        self.kwidths = kblocks.kwidths
        self.envs    = kblocks.envs
        self.kernels = kblocks.kernels

    # Method for applying kernels
    def tokenize(self, input):
        # Inefficient
        nbatch, hlen = input.shape
        input  = input.reshape(nbatch, 1, hlen)
        tokens = []

        # Apply kernels
        for i in range(0, self.nkers):
            # Pad to keep appropriate width 
            pad = int(self.kwidths[i]/2)-1

            # Convolution 
            tokens.append(torch.nn.functional.conv1d(
                input[:,:,-self.hlens[i]:], 
                self.kernels[i], 
                stride = self.strides[i], 
                padding = pad
                ).squeeze())

        return tokens


class FrameExpander(nn.Module):
    def __init__(self, kblocks, dim_hid):
        super().__init__()

        # Probably a better way to do this
        self.nkers   = kblocks.nkers
        self.hlens   = kblocks.hlens
        self.strides = kblocks.strides
        self.kwidths = kblocks.kwidths
        self.envs    = kblocks.envs
        self.kernels = kblocks.kernels

        # Output from transformer
        self.dim_hid = dim_hid

        # Layer for getting frame vector coefficients
        ceofs = torch.Linear(dim_hid, 500, bias=True)

    # Frame expansion
    def forward(self, input):

        # Frame coefficients
        y = self.coefs(input)

        # Aggregate
        z = torch.zeros(self.nkers)
        for i in range(0, self.nkers):
            z[:self.kwidths[i]] += y[i]*self.kernels[i]

        return z



def train(model, inputs, targs, lr = 0.001, nepochs = 2000):
    # Initialize the loss function and optimizer
    #loss_fn   = torch.nn.MSELoss()
    loss_fn   = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    
    # Get start time
    t0 = time()
        
    # Epoch loop
    for epoch in range(0, nepochs):
        
        # Reset gradient
        optimizer.zero_grad()
    
        # Model prediction
        pred = model(inputs)
        
        # Task and regularization losses
        loss = loss_fn(targs, pred)
        
        # Compute gradient
        loss.backward()
        
        # Apply gradient
        optimizer.step()
     
        # Say where we are in training
        if epoch % 10 == 0:
            loss, current = loss.item(), (epoch + 1) #* len(inputs)
            print(f"loss: {loss:>7f}  [{current:>5d}/{nepochs:>5d} max]")   
    
    # Print total elapsed time
    print('Training epochs: ' + str(epoch))
    print(f'Training time elapsed: {(time() - t0) / 60 :>7.2f}')
        

    

def get_actfn(type = 'identity'):

    if type == 'identity':
        return lambda x: x
    
    elif type == 'relu':
        return torch.nn.ReLU()
 
    elif type == 'tanh':
        return torch.tanh
    
    elif type == "softplus":
        return lambda x: torch.log(1 + torch.exp(2 * x)) / 2

    elif type == 'tanhplus':
        return lambda x: torch.nn.ReLU()(torch.tanh(3*x))

    elif type == 'tanhplusscaled':
        return lambda x,y: torch.nn.ReLU()(y*torch.tanh(2* 1/y *x))

    elif type == 'gelu':
        return torch.nn.GELU()
    
    else:
        error('Bad activation function.')
