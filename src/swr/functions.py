import pandas as pd
import os
import mne
import matplotlib
from scipy.io import savemat
from src.shared.presets import *

import scipy.io
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
import mne
import os
import scipy as sp
from scipy.signal import hilbert

import matplotlib.pyplot as plt
plt.ion()

def find_ca1_contacts(path_imaging):
    '''
    Read CA1 contacts for all subjs from imaging excel files.
    '''
    print('\nGetting contact info.')

    # Get all subjs in directory from dir names
    subjs = os.listdir(path_imaging)

    # Contacts dict with subj as key, timeseries as value
    contacts = dict()
    
    # Cycle over subjs
    for subj in subjs:
        try:
            # Make excel contact location file name
            imaging_file = path_imaging + subj + '/parsed.xlsx'

            # Notify user and read file
            print(f'Reading: {imaging_file}')
            data = pd.read_excel(imaging_file)

            # Get just the contacts that are in CA1
            contacts[subj] = list(data.loc[data['Location'] == 'CA1']['contact'])    

        # Handle file absence
        except FileNotFoundError:
            print(f"Failure: file not found")
        
        # Handle contact absence
        except KeyError:
            print(f"Failure: File lacks a column for contact Location")
    
    print('Completed.')
    return contacts


def check_contact_exists(contact_name, data_raw):
    # Possible alternate contact name: some contacts lost the hyphen
    alternate = contact_name[:1] + contact_name[2:] 

    # Figure out correct field name
    if contact_name in data_raw.ch_names:
        return True, contact_name
    elif alternate in data_raw.ch_names:
        return True, alternate
    else:
        return False, None


def copy_contact_data(chinfo, data_raw, contact):
    '''Checks contact exists, copies timeseries, creates filtered version.'''

    cdata = dict()
    channel_exists, name = check_contact_exists(contact, data_raw)
    if channel_exists:

        # Avoid filtering data needlessly
        drop = [name for name in data_raw.ch_names if name != name]
        l_freq = 80; h_freq = 100

        # Create copy and filtered copy
        cdata["act"   ], cdata["tvec"   ] = data_raw[name]
        cdata["act_bp"], cdata["tvec_bp"] = data_raw.copy().drop_channels(drop).filter(l_freq,h_freq,verbose='ERROR')[name]

        # Flatten activity channels. (Why aren't they already?)
        cdata['act'   ] = cdata['act'   ].flatten()
        cdata['act_bp'] = cdata['act_bp'].flatten()

        # Save channel metadata
        #cdata["contact_name"] = name
        #cdata["channel_info"] = chinfo[chinfo['contact'] == name].iloc[0].to_dict()

    return cdata
    

def get_ca1_data(paths, sessions, subjs, get_adj=True, save_mat=True):
    '''
    Extracts CA1 contacts and corresponding time series data from each subj's MNE "raw" files.
    
    Inputs:
        paths    : Class, paths.imaging and paths.raw_files point to imaging and mne data 
        sessions : List including Encoding, SameDayRecall, and/or NextDayRecall
        subjs    : List of subj ID strings
        get_adj  : Extracts TSD on electrode above and below CA1 on the lead

    Output: Nested dictionary of electrode and 
    '''
    print('\nGetting CA1 data from raw files...')

    # Preprocessed file suffixes
    preproc_sfx = '_no60hz_ref_bp_raw.fif'

    # Get contact information and subjs from imaging files
    contacts = find_ca1_contacts(paths.imaging)

    # Data format for copied data:
    lfps = {subj:{sess:{} for sess in sessions} for subj in subjs}

    # Loop through subjects, sessions, and contacts appending contct data
    for subj in subjs:
        for sess in sessions:
            # Say where we are
            print(f'Getting data for subj {subj}, session {sess}.')

            # Set current filename
            file = paths.raw_files + subj + '/' + sess + '/' + subj + '_' + sess + preproc_sfx
            
            # Check if file exists
            if not os.path.isfile(file):
                print(f'File does not exist: {file}')
                print('Skipping.')
                continue

            # Read file if it's there
            if file.endswith('.fif'):
                data_raw = mne.io.read_raw_fif(file, preload=True, verbose='ERROR')

            # get info for all channels for this subj
            channel_info_file = f"{paths.raw_files}{subj}/{subj}_chinfo.csv"
            print(f'Reading: {channel_info_file}')
            chinfo = pd.DataFrame(pd.read_csv(channel_info_file))
            chinfo.rename(columns = {'Level 3: gyrus/sulcus/cortex/nucleus': 'Level 3: subregion'}, inplace= True) # shorten for saving to matlab

            # go through channel contacts
            for contact in contacts[subj]:
                print(f'Processing contact: {contact}')

                # extract electrode names
                short_name  = contact[:1] + contact[2:] # some contacts lost the hyphen in their name
                contact_num = int(contact[-1:])
                contact_below = contact[:-1] + str(contact_num-1)
                contact_above = contact[:-1] + str(contact_num+1)

                # Copies timeseries data, returns {} if not in data_raw
                ca1_contact = copy_contact_data(chinfo, data_raw, contact)

                # Only save if data was found
                if len(ca1_contact) != 0:
                    lfps[subj][sess][short_name] = {'ca1_contact':ca1_contact}

                    # Add surrounding electrodes if requested
                    if get_adj:
                        lfps[subj][sess][short_name]["contact_below"] = copy_contact_data(chinfo, data_raw, contact_below)
                        lfps[subj][sess][short_name]["contact_above"] = copy_contact_data(chinfo, data_raw, contact_above)

    # Save data, maybe
    if save_mat: savemat('ca1_lfps.mat', {'lfps':lfps})

    return lfps

def get_trial_times(path, subj, sess, suffix):
    ''' Extracts start and stop times for every trial from MNE raw.fif file.'''

    # Generate filename
    file = path + subj + '/' + sess + '/' + subj + '_' + sess + suffix

    # Check if file is there, then read or skip
    if os.path.isfile(file):
        print(f'Reading file: {file}')
        raw = mne.io.read_raw_fif(file, preload=True, verbose='ERROR')
    else:
        print(f'No such file: {file}')
        print(f'Skipping.')
        return np.array([])

    # Get all annotated event tuples and dictionary of annotations
    array, dic = mne.events_from_annotations(raw, verbose='Error')

    # Find only the start and stop indices
    trial_beg_inds = np.where(array==dic['trial_start'])[0]
    trial_end_inds = np.where(array==dic['trial_stop'])[0]

    # Convert to times
    trial_beg = array[trial_beg_inds,0]
    trial_end = array[trial_end_inds,0]

    # Save start and end times
    return np.stack((trial_beg,trial_end),axis=-1)


def get_swr_trial_assoc(trial_start_times, subj, contact):
    '''Using SWR timing info, finds out which trials they're in.'''
    
    # Contact name according to matlab
    short_contact_name = contact[:1] + contact[2:]

    # Extract start times of SWRs. What on earth is this terrible, terrible data format?
    tstart = matfile_data_swrs['data'][subj][0][0][short_contact_name][0][0]['ca1_contact'][0][0]['tstart'][0][0]

    # Build a list with the trial number each SWR is in
    # Post-trial ITI is included in each trial except final
    swr_trial_assoc = []
    for time in tstart:

        # Get first index for trial start which is larger than swr start 
        ind = np.where(time < trial_start_times/1024)[0]
        
        # If an index was found, associate SWR with previous trial. Trails are 1-indexed.
        if not(len(ind) == 0):
            swr_trial_assoc.append(ind[0])
        else:
            # If no index found, it was after last trial
            swr_trial_assoc.append(trial_start_times.shape[0]+1)

    return np.array(swr_trial_assoc)


class SWRData:
    def __init__(self):
        self.idx_beg = None
        self.idx_end = None
        self.tbeg = None
        self.tend = None
        
        self.lfp_power_z = None
        self.n_events = 0

    def set(self, idx_beg, idx_end, tbeg, tend):
        self.idx_beg = idx_beg
        self.idx_end = idx_end
        self.tbeg = tbeg
        self.tend = tend

        self.n_events = idx_beg.shape[0]

    def apply_msk(self, msk):
        self.idx_beg = self.idx_beg[msk]
        self.idx_end = self.idx_end[msk]
        self.tbeg = self.tbeg[msk]
        self.tend = self.tend[msk]

        self.n_events = self.idx_beg.shape[0]


def get_swr_candidates(lfp, times, thresh_low=3, thresh_high=5, len_min=0.025, len_merge=0.05):
    '''This is LFPPower, TSDtoIV, AddTSDtoIV and SelectIV from the Van der Meer lab toolbox.'''
    # NB 2024-03-01: As of now, this function gives exactly the same answer as get_swrs_from_lfps.

    # Class for storing SWR information
    swrs = SWRData()

    # Get LFP power as z-scored envelope from hilbert transform
    swrs.lfp = lfp.flatten()
    swrs.lfp_power_z = sp.stats.zscore(abs(hilbert(lfp.flatten())))

    # Find super-threshold events
    super_thresh = (swrs.lfp_power_z > thresh_low).astype(int)
    
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
        max_val = np.max(swrs.lfp_power_z[idx:idx_end[i]])
        if max_val > thresh_high: keep.append(i)
    keep = np.array(keep)

    # Save data and drop events we don't want to keep
    swrs.set(idx_beg, idx_end, tbeg, tend)
    swrs.apply_msk(keep)

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

    return keep


def get_swrs_from_lfps(lfps):
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

                # LFP trace and time vector
                lfp   = lfps[subj][sess][contact]['ca1_contact']['act_bp'].flatten()
                times = lfps[subj][sess][contact]['ca1_contact']['tvec_bp'].flatten()
                
                # Get candidate SWR periods
                swrs = get_swr_candidates(lfp, times, thresh_high = 5)

                # Remove periods with elevated power on adjacent electrodes
                keep = check_adj_power(swrs.idx_beg, swrs.idx_end, lfps[subj][sess][contact], thresh=1)
                swrs.apply_msk(keep)

                # Save
                swrs_all[subj][sess][contact] = swrs

                # Check if desired
                if inspect_swrs:
                    for i in range(swrs.n_events):
                        ctx_beg = swrs.idx_beg[i] - int(1024/4)
                        ctx_end = swrs.idx_end[i] + int(1024/4)

                        ctxt = np.arange(ctx_beg, ctx_end)
                        iswr = np.arange(swrs.idx_beg[i], swrs.idx_end[i])

                        plt.figure(figsize=[8,2])
                        plt.plot(times[ctxt], swrs.lfp[ctxt])
                        plt.plot(times[iswr], swrs.lfp[iswr])

                        #plt.plot(times[ctxt], np.ones_like(ctxt)*3)
                        input('Press enter to continue')
                        plt.close('all')
    
    return swrs_all


##################################################
# The two functions below are in progress.
# I'm just pushing this now so I don't block anyone's progress.
##################################################
def create_swr_count_regressor():
    ### NOTICE this was a script and I haven't finished wrapping it to a function

    #file_suffix = '_Encoding_no60hz_ref_bp_raw.fif'
    #raw_path = "/oscar/data/brainstorm-ws/megagroup_data/"
    #path_matfiles = "seeg-replay/src/swr/"
    #matlab_pre = "ca1_data_matrix.mat"
    #matlab_post = "event_ca1_data.mat"
    #subjs = ['e0010GP', 'e0011XQ', 'e0013LW', 'e0015TJ', 'e0016YR', 'e0017MC']
    #sessions = ['Encoding']

    # Location of raw.fif mne files, and suffix for preprocessed files to use
    path_raw_files = "/home/dan/projects/work/megagroup_data/"
    file_suffix    = '_no60hz_ref_bp_raw.fif'

    # Location of contact and swr .mat files
    matfile_contacts = "./data/ca1_data_matrix.mat"
    matfile_swrs     = "./data/event_ca1_data.mat"

    # Subjects and sessions to process
    subjs    = ['e0010GP']#, 'e0011XQ', 'e0013LW', 'e0015TJ', 'e0016YR', 'e0017MC']
    sessions = ['Encoding']

    # Load the contact and SWR event info
    matfile_data_contacts = scipy.io.loadmat(matfile_contacts)
    matfile_data_swrs     = scipy.io.loadmat(matfile_swrs)

    # Get trial start times (indexing gets start column)
    trial_start_times = get_trial_times(path_raw_files, 'e0010GP', 'Encoding', file_suffix)[:,0]

    # Loop over subjects and sessions, 
    for subj, sess in zip(subjs,sessions):

        # Get usable contacts
        contacts = matfile_data_contacts['usable_contacts'][subj][0][0]
        
        # For each contact, associate SWRs with trials
        for contact in contacts:
            swr_trial_assoc = get_swr_trial_assoc(trial_start_times, subj, contact)


def aggregate_and_plot_swrs_by_performance():
    ###

    import matplotlib.pyplot as plt

    trial_swr_cnts = np.array([sum(swr_trial_assoc == trial) for trial in range(0,32)])

    cnt_data = pd.DataFrame(columns=['trial','condition','swr_cnt','error_position_encoding'])
    cnt_data['trial'] = np.arange(0,32)
    cnt_data['swr_cnt'] = trial_swr_cnts

    # Read behavior file
    behavior = pd.read_csv('./src/behavior/behavioral_data.csv')

    # Select subject, session for initializing condition numbers
    msk_subj = behavior['participant_id'] == 'e0010GP'
    msk_sess = behavior['session'] == 'Encoding'

    # Indexing in pandas .loc is inclusive. Will copy based on indices if .values is missing
    cnt_data.loc[1:30,'condition'] = behavior.loc[msk_subj & msk_sess]['condition'].values

    # Error data from the encoding session
    cnt_data.loc[1:30,'error_position_encoding'] = behavior.loc[msk_subj & msk_sess]['error_position'].values

    # Merge errors from SameDayRecall
    msk_sess = behavior['session'] == 'SameDayRecall'
    tmp_df = behavior.loc[msk_subj & msk_sess][['condition','error_position']].copy()
    tmp_df = tmp_df.rename(columns={'error_position':'error_position_same'})
    cnt_data = pd.merge(cnt_data, tmp_df, how = 'outer')

    # Merge errors form NextDayRecall
    msk_sess = behavior['session'] == 'NextDayRecall'
    tmp_df = behavior.loc[msk_subj & msk_sess][['condition','error_position']].copy()
    tmp_df = tmp_df.rename(columns={'error_position':'error_position_next'})
    cnt_data = pd.merge(cnt_data, tmp_df, how = 'outer')



    plt.figure()
    plt.plot(cnt_data['swr_cnt'], cnt_data['error_position_encoding'],'o')
    plt.xlabel('Num. SWRs')
    plt.ylabel('Position Error')
    plt.title('e0010GP SWR Count vs Encoding Position Accuracy')

    plt.figure()
    plt.plot(cnt_data['swr_cnt'], cnt_data['error_position_same'],'o')
    plt.xlabel('Num. SWRs while Encoding')
    plt.ylabel('Position Error')
    plt.title('e0010GP SWR Count vs SameDayRecall Position Accuracy')

    plt.figure()
    plt.plot(cnt_data['swr_cnt'], cnt_data['error_position_next'],'o')
    plt.xlabel('Num. SWRs while Encoding')
    plt.ylabel('Position Error')
    plt.title('e0010GP SWR Count vs NextDayRecall Position Accuracy')

    plt.figure()
    plt.plot(cnt_data['condition'], cnt_data['swr_cnt'],'o')
    plt.xlabel('Condition Number')
    plt.ylabel('SWR Count')
    plt.title('e0010GP Encoding SWR Count by Condition')

    plt.figure()
    plt.plot(cnt_data['trial'], cnt_data['swr_cnt'],'o')
    plt.xlabel('Trial Number')
    plt.ylabel('SWR Count')
    plt.title('e0010GP Encoding SWR Count by Trial')

