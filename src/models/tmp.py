def old():


    # Get data
    #dps  = DataParams()
    #data = DataClass(dps)

    # Chunk it
    #inputs, targs, dps.nclps = chunk_data(dps, data)

    # Held out fraction, max clip index for training
    #hof    = 0.2
    #cutoff = int(np.floor(dps.nclps*(1-hof)))

    # Model dimensions
    dim_in  = dps.w_len_hist * dps.nch
    dim_out = dps.w_len_pred
    dim_emb = 64
    nhead   = 4
    nlayers = 6

    # Initialize network
    net = models.Transformer(dim_in, dim_out, dim_emb, nhead, nlayers)

    # Make sure everything is on the same device
    to_device(inputs, targs, net)

    # Train the model
    inputs = inputs[0:100,:]
    targs  = targs[0:100,:]





    loc = 'precentral gyrus'

    x_ctct = loc_dict[loc][0]
    y_ctct = loc_dict[loc][1]
    x = data[x_ctct][0].flatten()
    y = data[y_ctct][0].flatten()

    times = data[x_ctct][1].flatten()




    plt.plot(times, x)

    for t1,t2 in zip(code_t['clip_start'],code_t['clip_stop']):
        plt.axvspan(t1, t2, alpha=0.2, color='red')

    for t1,t2 in zip(code_t['clipcue_start'],code_t['clipcue_stop']):
        plt.axvspan(t1, t2, alpha=0.2, color='green')

    for t1,t2 in zip(code_t['loc_start'],code_t['loc_resp']):
        plt.axvspan(t1, t2, alpha=0.2, color='magenta')
