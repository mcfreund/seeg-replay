#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 15:28:22 2023

@author: dan
"""
from time  import time

import torch
from torch import nn, Tensor

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
