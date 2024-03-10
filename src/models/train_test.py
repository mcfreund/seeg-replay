import torch
from time  import time
import numpy as np
import matplotlib.pyplot as plt

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
        

#
def predict(net, inputs, targs, dps, plot = True):

    # Training predictions for checking accuracy
    pred = net(inputs).cpu().detach().numpy()

    # Re-aggregate them into single series
    nprds = pred.shape[0]
    npts  = dps.w_len_pred + dps.stride*(nprds - 1)
    agg   = aggregate_predictions(npts, nprds, dps.stride, pred)
    trg   = aggregate_predictions(npts, nprds, dps.stride, targs.cpu().detach().numpy())

    # Figures
    if plot:
        # Plot Training
        plt.figure()
        plt.plot(agg, alpha = 0.6)
        plt.plot(trg, alpha = 0.6)
        plt.title('Predictions')

        # Correlations
        cc = np.zeros(nprds)
        for i in range(nprds):
            cc[i] = np.corrcoef(pred[i,:], targs[i,:].cpu().detach().numpy())[0,1]

        plt.figure()
        plt.plot(cc)
        plt.title('Corrs')

    return agg, trg

    
def aggregate_predictions(npts, nprds, stride, pred):

    # How far out to aggregate over? (min should be stride)
    fwd_samples = 10

    # Aggregate prediction
    agg = np.zeros(npts)
    cnt = np.zeros(npts)

    # Loop through predictions
    cntvec = np.ones(fwd_samples)
    for i in range(nprds):
        
        # Determine where this prediction began and ended
        beg = stride*i
        end = stride*i + fwd_samples

        # Add this prediction to the correct aggregation 
        agg[beg:end] += pred[i,0:fwd_samples]
        cnt[beg:end] += cntvec

    # Normalize
    for i in range(npts):
        agg[i] = agg[i]/cnt[i] if cnt[i] > 0 else agg[i]

    # Return
    return agg
