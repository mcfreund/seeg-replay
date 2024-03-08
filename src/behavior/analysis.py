import pandas as pd
import numpy  as np
from matplotlib  import pyplot as plt
from scipy.stats import gaussian_kde
plt.ion()

### Settings
debug     = False
save_figs = True



###

# Load data
data  = pd.read_csv("/oscar/data/brainstorm-ws/megagroup_data/behavioral_data.csv")

# Subject info
sids  = data['participant_id'].unique()
nsubj = len(sids)

# Session info
sess_ids = data['session'].unique()

# Debugging
if debug: sids = [sids[0]]

# Cycle through subjects
for snum, subj in enumerate(sids):

    #plt.figure(figsize = [8,6])
    plt.subplots(3,2, figsize=[8,6], gridspec_kw={'width_ratios': [5, 1]})

    # Cycle through sessions
    for i, sess in enumerate(sess_ids):
        # Masks for the data
        msk_subj = data.participant_id == subj
        msk_sess = data.session == sess

        # The data itself
        trials  = data.loc[msk_subj & msk_sess, 'trial_num']
        err_pos = abs( data.loc[msk_subj & msk_sess,'error_position'])
        err_col = abs( data.loc[msk_subj & msk_sess,'error_colorpos'   ])

        # Data length
        dlen = np.shape(err_pos)[0]

        # Compute KDEs
        kde_pos  = gaussian_kde(err_pos)
        kde_col  = gaussian_kde(err_col)
        support  = np.linspace(0, 180, 1000)
        pos_vals = kde_pos.evaluate(support)
        col_vals = kde_col.evaluate(support)

        # Get 25th, 50th, 75th percentiles
        cdf = np.cumsum(pos_vals)
        cdf = cdf / cdf[-1]

        # Get percentiles, append endpoints
        prct = np.interp([0.25, 0.50, 0.75], cdf, support)
        prct = [round(i) for i in prct]
        prct = [0, *prct, 180]

        # Find the maximum value on the KDE curves
        max_kde_value = np.max([pos_vals.max(), col_vals.max()])

        # Plot subject performance
        plt.subplot(3, 2, (i*2)+1)
        plt.plot( trials, err_pos, '-o')
        plt.plot( trials, err_col, '-o')
        plt.ylim([0,180])
        plt.yticks([0, 45, 90, 135, 180])
        plt.title( 'Subject ' + str(snum) + ' ' + sess + ' Performance')
        plt.legend(['position','color'])
        plt.grid()

        # KDE subplots
        plt.subplot(3,2, (i*2) + 2)

        # Plot percentiles of KDE
        for i in range(1,len(prct)):
            msk = (support > prct[i-1]) & (support < prct[i])
            plt.plot(support[msk], pos_vals[msk])
        
        # Plot rug plot of errors
        plt.plot(err_pos, np.random.rand(dlen)*0.10*max_kde_value, '.')
        plt.xlim([0, 180])
        plt.ylim([0, max_kde_value*1.05])

        # Labels for the KDE plot
        # plt.legend(['kde(0,25)', 'kde(25,50)', 'kde(50,75)', 'kde(75,100)','obs'])
        plt.title('KDE(PErr)')

        # Fix up layout
        plt.tight_layout()
    
    # Save outside subplot loop
    if save_figs: plt.savefig('../../figs/behavior/performance-subject-' + str(snum) + '-' + str(subj) + '.png')


