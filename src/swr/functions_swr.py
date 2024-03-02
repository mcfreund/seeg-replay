import os, mne
import numpy  as np
import pandas as pd
import scipy  as sp
import matplotlib.pyplot as plt
from scipy.signal  import hilbert
from src.shared.utils import *
plt.ion()

class SWRData:
    def __init__(self):
        '''Packaging for SWR information, including timing and SWR LFPs'''

        # Indexing and timing information
        self.idx_beg = None
        self.idx_end = None
        self.tbeg  = None
        self.tend  = None
        
        # Event count
        self.n_events = 0

        # Actual cuts from LFP
        self.times     = []
        self.traces    = []
        self.traces_bp = []

    def save(self, idx_beg, idx_end, tbeg, tend, times, lfp, lfp_bp):
        '''Writes object data. Over-writes everything.'''
        self.idx_beg = idx_beg
        self.idx_end = idx_end
        self.tbeg = tbeg
        self.tend = tend

        self.n_events = idx_beg.shape[0]

        # Re-create traces lists
        self.times     = []
        self.traces    = []
        self.traces_bp = []
        for i in range(self.n_events):
            self.times.append(     times[ idx_beg[i]:idx_end[i] ])
            self.traces.append(      lfp[ idx_beg[i]:idx_end[i] ])
            self.traces_bp.append(lfp_bp[ idx_beg[i]:idx_end[i] ])

    def keep_inds(self, inds):
        '''Subsets the data (drops SWRs)'''
        self.idx_beg = self.idx_beg[inds]
        self.idx_end = self.idx_end[inds]
        self.tbeg = self.tbeg[inds]
        self.tend = self.tend[inds]

        self.n_events = self.idx_beg.shape[0]

        self.times     = [self.times[i]     for i in inds]
        self.traces    = [self.traces[i]    for i in inds]
        self.traces_bp = [self.traces_bp[i] for i in inds]


def get_swr_candidates(lfp, lfp_bp, times, thresh_low=3, thresh_high=5, len_min=0.025, len_merge=0.05):
    '''This is LFPPower, TSDtoIV, AddTSDtoIV and SelectIV from the Van der Meer lab toolbox.'''
    # NB 2024-03-01: As of now, this function gives exactly the same answer as get_swrs_from_lfps.

    # Class for storing SWR information
    swrs = SWRData()

    # Get LFP power as z-scored envelope from hilbert transform
    lfp_power_z = sp.stats.zscore(abs(hilbert(lfp_bp.flatten())))

    # Find super-threshold events
    super_thresh = (lfp_power_z > thresh_low).astype(int)
    
    # Indices and times for super-threshold events
    dfs      = np.diff(super_thresh)
    idx_beg  = np.where(dfs == 1)[0]    +1   # Indexing correction rel to matlab
    idx_end  = np.where(dfs == -1)[0]-1 +1   # Indexing correction rel to matlab
    time_beg = times[idx_beg]
    time_end = times[idx_end]

    # Find locations that should be merged
    d = time_beg[1:] - time_end[0:-1]
    idx_merge = np.where(d < len_merge)[0]
    
    # Remove up and down times for merged items
    idx_keep_beg = np.setdiff1d(np.arange(len(idx_beg)),idx_merge+1)
    idx_keep_end = np.setdiff1d(np.arange(len(idx_beg)),idx_merge  )
    time_beg = time_beg[idx_keep_beg]
    time_end = time_end[idx_keep_end]
    idx_beg  = idx_beg[ idx_keep_beg]
    idx_end  = idx_end[ idx_keep_end]

    # Get interval lengths, keep only long enough
    ivlens   = time_end - time_beg
    keep_idx = ivlens > len_min
    tbeg     = time_beg[keep_idx]
    tend     = time_end[keep_idx]
    idx_beg  = idx_beg[ keep_idx]
    idx_end  = idx_end[ keep_idx]

    # Filter candidate SWR periods by max deviation value
    keep = []
    for i, idx in enumerate(idx_beg):
        max_val = np.max(lfp_power_z[idx:idx_end[i]])
        if max_val > thresh_high: keep.append(i)
    keep = np.array(keep)

    # Drop events we don't want to keep, save data
    swrs.save(idx_beg, idx_end, tbeg, tend, times, lfp, lfp_bp)
    swrs.keep_inds(keep)

    return swrs


def check_adj_power(idx_beg, idx_end, contact_dict, thresh=1):
    '''Check adjacent electrode power to minimize false positive SWR detections.'''

    # Get adjacent electrode data
    adj_list = [contact_dict['contact_above'], contact_dict['contact_below']]

    # Mask for keeping indices
    n_events = len(idx_beg)
    keep = np.array([True]*n_events)

    # Check if each is empty, and if not, process.
    for adj in adj_list:
        if adj != {}:
            # Get the adjacent LFP data
            lfp_adj = adj['act_bp'].flatten()

            # Get power on this adjacent electrode
            adj_power_z = sp.stats.zscore(abs(hilbert(lfp_adj.flatten())))

            # Check each swr event 
            for i in range(n_events):

                # See if reference electrode is elevated
                if np.mean(adj_power_z[idx_beg[i]:idx_end[i]]) > thresh:
                    keep[i] = False

    return np.where(keep)[0]


def get_swrs_from_lfps(lfps, save_pkl = True):
    '''Gets SWR event times from LFP data.'''
    # NB this is confirmed as of 2024-03-01 to produce the exact same output as get_swrs_from_lfps.m
    # But without all the matlab nonsense.
    inspect_swrs = False

    # Initialize SWR collection dict
    swrs_all = {subj:{sess:{contact:{} for contact in lfps[subj][sess]} for sess in lfps[subj]} for subj in lfps}
    
    # Loop over subjects, sessions, contacts, detecting SWRs and saving them
    for subj in lfps:
        for sess in lfps[subj]:
            for contact in lfps[subj][sess]:
                # Notify where we are in loop
                print(f'Getting SWRs for subj: {subj}   session: {sess: <13}   contact: {contact}')

                # LFP trace and time vector
                lfp    = lfps[subj][sess][contact]['ca1_contact']['act'    ].flatten()
                lfp_bp = lfps[subj][sess][contact]['ca1_contact']['act_bp' ].flatten()
                times  = lfps[subj][sess][contact]['ca1_contact']['tvec_bp'].flatten()
                
                # Get candidate SWR periods
                swrs = get_swr_candidates(lfp, lfp_bp, times, thresh_high = 5)

                # Remove SWRs with elevated power on adjacent electrodes
                keep = check_adj_power(swrs.idx_beg, swrs.idx_end, lfps[subj][sess][contact], thresh=1)
                swrs.keep_inds(keep)

                # Save
                swrs_all[subj][sess][contact] = swrs

                # Visually inspect if desired
                if inspect_swrs:
                    for i in range(swrs.n_events):
                        ctx_beg = swrs.idx_beg[i] - int(1024/4)
                        ctx_end = swrs.idx_end[i] + int(1024/4)

                        ctxt = np.arange(ctx_beg, ctx_end)
                        iswr = np.arange(swrs.idx_beg[i], swrs.idx_end[i])

                        plt.figure(figsize=[8,2])
                        plt.plot(times[ctxt], swrs.lfp[ctxt])
                        plt.plot(times[iswr], swrs.lfp[iswr])

                        input('Press enter to continue')
                        plt.close('all')
    
    # Indicate how many there were
    #print(f'Candidate SWRs found: {swrs.n_events}, number spurious: {sum(~keep)}, kept: {sum(keep)}')

    # Save if requested
    if save_pkl: dill_save(swrs_all, './data/ca1_swrs.pt')

    return swrs_all


def append_swr_trial_assoc(swrs, paths, subjs, sessions):
    '''Associates SWRs with trials. Appends info to swr.'''

    # Get trial start and end times
    print('Getting trial info...')
    trial_times = get_trial_times_(paths, subjs, sessions)

    # Get association for each SWR by subject, session, contact
    for subj in subjs:
        for sess in sessions:
            for contact in swrs[subj][sess]:

                # Say where we are in case of failure or latency
                print(f'Finding SWR trials for {subj}, {sess: <13}, {contact}')

                # Get number of trials
                n_trials = len(trial_times[subj][sess][:,0])

                # Build a list with the trial number each SWR is in
                trials = []
                for start_time in swrs[subj][sess][contact].tbeg:

                    # Get first index for trial start which is larger than swr start 
                    ind = np.where(start_time < trial_times[subj][sess][:,0]/1024)[0]
                    
                    # If an index was found, associate SWR with previous trial. Trails are 1-indexed.
                    if not(len(ind) == 0):
                        trials.append(ind[0])
                    else:
                        # If no index found, it was after last trial
                        trials.append(trial_times[subj][sess].shape[0]+1)

                # Append this list to SWR data
                trials = np.array(trials)
                swrs[subj][sess][contact].n_trials = n_trials
                swrs[subj][sess][contact].trials   = trials
                swrs[subj][sess][contact].trial_counts = np.array([sum(trials == trial) for trial in range(0,n_trials+2)])

    return swrs



def aggregate_and_plot_swrs_by_performance(swrs):
    ###

    import matplotlib.pyplot as plt

    # Frame for collecting behavior with SWR data
    df = pd.DataFrame(columns=['trial','condition','swr_cnt_enc','err_pos_enc'])

    # Read behavior file
    behavior = pd.read_csv('./src/behavior/behavioral_data.csv')

    #
    subj = 'e0010GP'
    sess = 'Encoding'
    ctct = 'CMHIP2'

    df['trial'  ] = np.arange(0, swrs[subj][sess][ctct].n_trials + 2)
    df['swr_cnt_enc'] = swrs[subj][sess][ctct].trial_counts
    
    # Select subject, session for initializing condition numbers
    msk_subj = behavior['participant_id'] == subj
    msk_sess = behavior['session']        == sess

    # Copy behavioral data. Indexing in pandas .loc is inclusive.
    df.loc[1:swrs[subj][sess][ctct].n_trials,'condition']   = behavior.loc[msk_subj & msk_sess]['condition'].values
    df.loc[1:swrs[subj][sess][ctct].n_trials,'err_pos_enc'] = behavior.loc[msk_subj & msk_sess]['error_position'].values

    # Merge errors from SameDayRecall
    msk_sess = behavior['session'] == 'SameDayRecall'
    tmp_df = behavior.loc[msk_subj & msk_sess][['condition','error_position']].copy()
    tmp_df = tmp_df.rename(columns={'error_position':'err_pos_same'})
    df = pd.merge(df, tmp_df, how = 'outer')

    # Merge errors from NextDayRecall
    msk_sess = behavior['session'] == 'NextDayRecall'
    tmp_df = behavior.loc[msk_subj & msk_sess][['condition','error_position']].copy()
    tmp_df = tmp_df.rename(columns={'error_position':'err_pos_next'})
    df = pd.merge(df, tmp_df, how = 'outer')



    # Merge swrs from SameDayRecall
    msk_sess = behavior['session'] == 'SameDayRecall'
    tmp_df = behavior.loc[msk_subj & msk_sess][['condition']].copy()
    tmp_df['swr_cnt_same'] = swrs[subj]['SameDayRecall'][ctct].trial_counts[1:-1]
    df = pd.merge(df, tmp_df, how = 'outer')

    # Merge swrs from NextDayRecall
    msk_sess = behavior['session'] == 'NextDayRecall'
    tmp_df = behavior.loc[msk_subj & msk_sess][['condition']].copy()
    tmp_df['swr_cnt_next'] = swrs[subj]['NextDayRecall'][ctct].trial_counts[1:-1]
    df = pd.merge(df, tmp_df, how = 'outer')


    # Encoding counts vs performance
    plt.figure()
    plt.plot(df['swr_cnt_enc'], df['err_pos_enc'],'o')
    plt.xlabel('Num. SWRs'); plt.ylabel('Position Error')
    plt.title('e0010GP Encoding SWR by Encoding Accuracy')

    plt.figure()
    plt.plot(df['swr_cnt_enc'], df['err_pos_same'],'o')
    plt.xlabel('Num. SWRs'); plt.ylabel('Position Error')
    plt.title('e0010GP Encoding SWR by SameDay Accuracy')

    plt.figure()
    plt.plot(df['swr_cnt_enc'], df['err_pos_next'],'o')
    plt.xlabel('Num. SWRs'); plt.ylabel('Position Error')
    plt.title('e0010GP Encoding SWR by NextDay Accuracy')

    # Same-day counts vs performnace
    plt.figure()
    plt.plot(df['swr_cnt_same'], df['err_pos_same'],'o')
    plt.xlabel('Num. SWRs'); plt.ylabel('Position Error')
    plt.title('e0010GP SameDay SWR by SameDay Accuracy')

    plt.figure()
    plt.plot(df['swr_cnt_same'], df['err_pos_next'],'o')
    plt.xlabel('Num. SWRs'); plt.ylabel('Position Error')
    plt.title('e0010GP SameDay SWR by NextDay Accuracy')

    # Next-day counts vs performance
    plt.figure()
    plt.plot(df['swr_cnt_next'], df['err_pos_next'],'o')
    plt.xlabel('Num. SWRs'); plt.ylabel('Position Error')
    plt.title('e0010GP NextDay SWR by NextDay Accuracy')


    # SWR counts by trial and condition
    plt.figure()
    plt.plot(df['trial'], df['swr_cnt_enc'],'o')
    plt.xlabel('Trial Number')
    plt.ylabel('SWR Count')
    plt.title('e0010GP Encoding SWR Count by Trial')

    plt.figure()
    plt.plot(df['condition'], df['swr_cnt_enc'],'o')
    plt.xlabel('Condition Number')
    plt.ylabel('SWR Count')
    plt.title('e0010GP Encoding SWR Count by Condition')



