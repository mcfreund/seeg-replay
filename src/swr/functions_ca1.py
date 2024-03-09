import os, mne
import pandas as pd
from scipy.io import savemat
import dill

from src.shared.paths import *
from src.shared.utils import *

def copy_contact_data(chinfo, data_raw, contact):
    '''Checks contact exists, copies timeseries, creates filtered version.'''
    cdata = dict()
    channel_exists, name = check_contact_exists(contact, data_raw)
    if channel_exists:
        # Avoid filtering channels we don't want
        drop = [name for name in data_raw.ch_names if name != name]
        l_freq = 80; h_freq = 100

        # Create copy and filtered copy
        cdata["act"   ], cdata["tvec"   ] = data_raw[name]
        cdata["act_bp"], cdata["tvec_bp"] = data_raw.copy().drop_channels(drop).filter(l_freq, h_freq, verbose='ERROR')[name]

        # Flatten activity channels. (Why aren't they already?)
        cdata['act'   ] = cdata['act'   ].flatten()
        cdata['act_bp'] = cdata['act_bp'].flatten()

    return cdata


def get_lfp_data(paths, sessions, subjs, loc='CA1', get_adj=True, save_mat=False, save_pkl=True):
    '''
    Extracts specified region's timeseries from each subj's MNE "raw" files.
    Inputs:
        paths    : Class, paths.imaging and paths.processed_raws point to imaging and mne data 
        sessions : List including Encoding, SameDayRecall, and/or NextDayRecall
        subjs    : List of subj ID strings
        loc      : Contact location to check for (CA1 is legacy behavior)
        get_adj  : Extracts TSD on electrode above and below CA1 on the lead
        save_mat : Save matlab file? Required if running LFPeventDetection_EEG next
        save_pkl : Save pickle file? Useful for skipping this in python processing pipeline.
    Output: 
        Nested dictionary such as lfps[subj][sess][contact][ca1_contact]. See use of copy_contact_data().
    '''
    print('\nGetting CA1 data from raw files...')

    # Get contact information and subjs from imaging files
    contacts = find_contacts(paths, subjs, loc=loc)

    # Data format for copied data:
    lfps = {subj:{sess:{} for sess in sessions} for subj in subjs}

    # Loop through subjects, sessions, and contacts appending contct data
    for subj in subjs:
        for sess in sessions:
            # Say where we are
            print(f'Getting data for subj {subj}, session {sess}.')

            # Set current filename
            file = paths.processed_raws + subj + '/' + sess + '/' + subj + '_' + sess + paths.suffix
            
            # Check if file exists
            if not os.path.isfile(file):
                print(f'File does not exist: {file}')
                continue

            # Read file if it's there
            if file.endswith('.fif'):
                data_raw = mne.io.read_raw_fif(file, preload=True, verbose='ERROR')

            # get info for all channels for this subj
            channel_info_file = f"{paths.processed_raws}{subj}/{subj}_chinfo.csv"
            print(f'Reading: {channel_info_file}')
            chinfo = pd.DataFrame(pd.read_csv(channel_info_file))
            chinfo.rename(columns = {'Level 3: gyrus/sulcus/cortex/nucleus': 'Level 3: subregion'}, inplace= True) # shorten for saving to matlab

            # go through channel contacts
            for contact in contacts[subj]:
                print(f'Processing contact: {contact}')

                # extract electrode names
                short_name  = contact[:1] + contact[2:] # some contacts lost the hyphen in their name
                contact_above, contact_below = get_adjacent_contacts(contact)

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
    if save_pkl: dill_save(lfps, './data/ca1_lfps.pt')

    return lfps