import mne
import numpy  as np
import pandas as pd
import scipy  as sp
import matplotlib.pyplot as plt

from scipy.signal import hilbert
from datetime     import datetime

from src.shared.utils  import *
from src.swr.plots     import *

class SWRData:
    def __init__(self):
        '''Packaging for SWR information, including timing and SWR LFPs'''

        # Indexing and timing information
        self.idx_beg = None  # SWR     start indices into LFP and time vectors
        self.idx_end = None  # SWR     end   indices into LFP and time vectors
        self.ctx_beg = None  # Context start indices into LFP and time vectors
        self.ctx_end = None  # Context end   indices into LFP and time vectors
        self.tbeg = None     # Actual SWR beginning time
        self.tend = None     # Actual SWR end

        # Event count
        self.n_events = 0

        # Actual cuts from LFP
        self.times     = []
        self.ctime     = []
        self.ctxts     = []
        self.traces    = []
        self.traces_bp = []

    def save(self, idx_beg, idx_end, tbeg, tend, times, lfp, lfp_bp):
        '''Writes object data. Over-writes everything.'''
        self.idx_beg = idx_beg
        self.idx_end = idx_end
        self.tbeg = tbeg
        self.tend = tend

        self.ctx_beg = np.array([idx - int(1024/4) for idx in self.idx_beg])
        self.ctx_end = np.array([idx + int(1024/4) for idx in self.idx_end])

        self.n_events = idx_beg.shape[0]

        # Re-create traces lists
        self.times     = []
        self.ctime     = []
        self.ctxts     = []
        self.traces    = []
        self.traces_bp = []
        for i in range(self.n_events):
            self.times.append(     times[ idx_beg[i]:idx_end[i] ])
            self.traces.append(      lfp[ idx_beg[i]:idx_end[i] ])
            self.traces_bp.append(lfp_bp[ idx_beg[i]:idx_end[i] ])

            self.ctime.append( times[ self.ctx_beg[i]:self.ctx_end[i] ])
            self.ctxts.append(   lfp[ self.ctx_beg[i]:self.ctx_end[i] ])

    def keep_inds(self, inds):
        '''Subsets the data (drops SWRs)'''
        # Avoid subsetting with empty numpy array
        if len(inds) == 0:
            inds = []

        # Subset everything
        self.idx_beg = self.idx_beg[inds]
        self.idx_end = self.idx_end[inds]
        self.ctx_beg = self.ctx_beg[inds]
        self.ctx_end = self.ctx_end[inds]
        
        self.tbeg = self.tbeg[inds]
        self.tend = self.tend[inds]

        self.n_events = self.idx_beg.shape[0]

        self.times     = [self.times[i]     for i in inds]
        self.ctime     = [self.ctime[i]     for i in inds]
        self.ctxts     = [self.ctxts[i]     for i in inds]
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
    super_thresh = np.concatenate([[0], super_thresh, [0]])

    # Indices and times for super-threshold events
    dfs      = np.diff(super_thresh)
    idx_beg  = np.where(dfs ==  1)[0]  
    idx_end  = np.where(dfs == -1)[0]-1
    time_beg = times[idx_beg]
    time_end = times[idx_end]

    #ipdb.set_trace()
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


def check_adj_power(idx_beg, idx_end, contact_dict, thresh=1, raws=None, use_raws=False):
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


def check_adj_power_raws(idx_beg, idx_end, raw, contact, thresh=1):
    '''Check adjacent electrode power to minimize false positive SWR detections.'''

    # Contacts numbered +1 and -1 in the same target region
    contact_above, contact_below = get_adjacent_contacts(contact)

    # Check which adjacent contacts exist
    above_exists, contact_above = check_contact_exists(contact_above, raw)
    below_exists, contact_below = check_contact_exists(contact_below, raw)

    # Get adjacent electrode data
    adj_exist = [above_exists , below_exists ]
    adj_names = [contact_above, contact_below]

    # Mask for keeping indices
    n_events = len(idx_beg)
    keep = np.array([True]*n_events)

    # Check if each is empty, and if not, process.
    for i in range(2):
        if adj_exist:
            # Get the adjacent LFP data
            adj_power_z = raw[adj_names[i]][0].flatten()

            # Check each swr event 
            for i in range(n_events):

                # See if reference electrode is elevated
                if np.mean(adj_power_z[idx_beg[i]:idx_end[i]]) > thresh:
                    keep[i] = False

    return np.where(keep)[0]


def get_swrs_from_lfps(lfps, save_pkl=False, save_swrs=False):
    '''Gets SWR event times from LFP data.'''
    # NB this is confirmed as of 2024-03-01 to produce the exact same output as get_swrs_from_lfps.m

    # Create folder to save things in if requested
    if save_swrs:
        datestr  = datetime.now().strftime('%y-%m-%d--%H-%M-%S')
        dir_figs = './figs/' + datestr + '-swrs-ca1'
        os.mkdir(dir_figs)

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
                #swrs.keep_inds(keep)

                # Indicate how many there were
                #print(f'Candidate SWRs found: {swrs.n_events}, kept: {len(keep)}')

                # Indicate how many there were
                print(f'Candidate SWRs found: {swrs.n_events}')

                # Save
                swrs_all[subj][sess][contact] = swrs

                # Visually inspect if desired
                if save_swrs:
                    # For plotting:
                    pow_z = sp.stats.zscore(abs(hilbert(lfp_bp.flatten())))
                    for i in range(swrs.n_events):
                        plot_swr(i, swrs, lfp_bp, pow_z, contact)
                        plt.savefig(f'{dir_figs}/{subj}_{contact}_{sess}_swr_{i}.png')
                        plt.close()
    
    # Save if requested
    if save_pkl: dill_save(swrs_all, './data/ca1_swrs_' + datestr + '.pt')

    return swrs_all



def get_swrs_from_raws(paths, subjs, sessions, save_pkl = False, save_swrs = False, inspect_swrs = False):
    '''Gets SWR event times from LFP data.'''

    # Create folder to save things in if requested
    if save_swrs:
        datestr  = datetime.now().strftime('%y-%m-%d--%H-%M-%S')
        dir_figs = './figs/' + datestr + '-swrs'
        os.mkdir(dir_figs)

    # Initialize SWR collection dict
    swrs_all = {subj:{sess:{} for sess in sessions} for subj in subjs}
    
    # Loop over subjects, sessions, contacts, detecting SWRs and saving them
    for subj in subjs:
        for sess in sessions:

            # Load preprocessed raw file for LFPs and LFP z-scored power file
            file_lfp = paths.processed_raws + subj + '/' + sess + '/' + subj + '_' + sess + paths.suffix
            file_bp  = paths.processed_raws + subj + '/' + sess + '/' + subj + '_' + sess + '_lfp_80_100_bp_raw.fif'
            file_pow = paths.processed_raws + subj + '/' + sess + '/' + subj + '_' + sess + '_lfp_power_z_raw.fif'
            raw_lfp  = mne.io.read_raw_fif(file_lfp, preload = True, verbose = 'Error')
            raw_bp   = mne.io.read_raw_fif(file_bp , preload = True, verbose = 'Error')
            raw_pow  = mne.io.read_raw_fif(file_pow, preload = True, verbose = 'Error')

            # Get white matter contacts to drop.
            wmctcts = find_contacts(paths, subjs, loc='white matter')[subj]

            # Drop them.
            raw_lfp = raw_lfp.drop_channels(wmctcts, on_missing='ignore')
            raw_bp  = raw_bp.drop_channels(wmctcts, on_missing='ignore')
            raw_pow = raw_pow.drop_channels(wmctcts, on_missing='ignore')

            # Numpy arrays of LFP data, power data, time
            lfp_array = raw_lfp._data
            bp_array  = raw_bp._data
            pow_array = raw_pow._data
            times     = raw_lfp['data'][1]

            # Check every contact for SWRs
            for i, contact in enumerate(raw_lfp.ch_names):

                # Notify where we are in loop
                print(f'Getting SWRs for subj: {subj}   session: {sess: <13}   contact: {contact}')

                # Get candidate SWR periods
                swrs = get_swr_candidates(lfp_array[i], pow_array[i], times, thresh_high = 5)

                # Remove SWRs with elevated power on adjacent electrodes
                keep = check_adj_power_raws(swrs.idx_beg, swrs.idx_end, raw_pow, contact, thresh=1)
                swrs.keep_inds(keep)

                # Indicate how many there were
                print(f'Candidate SWRs found: {swrs.n_events}') #, number spurious: {sum(~keep)}, kept: {sum(keep)}')

                # Save
                swrs_all[subj][sess][contact] = swrs

                # Visually inspect if desired
                if inspect_swrs:
                    plt.ion()
                    for j in range(swrs.n_events):
                        plot_swr(j, swrs, bp_array[i], pow_array[i], contact)
                        input('Press enter to continue')
                        plt.close('all')
    
                if save_swrs:
                    # For plotting:
                    for j in range(swrs.n_events):
                        plot_swr(j, swrs, bp_array[i], pow_array[i], contact)
                        plt.savefig(f'{dir_figs}/{subj}_{contact}_{sess}_swr_{i}.png')
                        plt.close()
    
    # Save if requested
    if save_pkl: dill_save(swrs_all, './data/ca1_swrs_from_raws.pt')

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



def aggregate_swrs_and_behavior(swrs):
    ###

    import matplotlib.pyplot as plt

    # Frame for collecting behavior with SWR data
    df = pd.DataFrame(columns=['trial','condition','swr_cnt_enc','err_pos_enc'])

    # Read behavior file
    behavior = pd.read_csv('./data/behavioral_data.csv')

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

    return df



def get_swr_bp_lfp_power_z(paths, subjs, sessions):
    '''
    '''

    # Loop through subjects, sessions, and contacts appending contct data
    for subj in subjs:
        for sess in sessions:
            # Say where we are
            print(f'Getting data for subj {subj}, session {sess}.')

            # Set current filename
            file_read  = paths.processed_raws + subj + '/' + sess + '/' + subj + '_' + sess + paths.suffix
            file_write = paths.processed_raws + subj + '/' + sess + '/' + subj + '_' + sess
            
            # Read raw file
            raw = mne.io.read_raw_fif(file_read, preload = True)

            # Bandpass filter data and save
            low_freq =  80; high_freq = 100
            raw = raw.filter(low_freq, high_freq)
            raw.save(file_write + '_lfp_80_100_bp_raw.fif', overwrite = True)

            # Apply Hilbert transform and z-score
            raw = raw.apply_hilbert(envelope = True)
            for i in range(raw['data'][0].shape[0]):
                raw._data[i] = sp.stats.zscore(raw['data'][0][i,:])

            # Save
            raw.save(file_write + '_lfp_power_z_raw.fif', overwrite = True)
