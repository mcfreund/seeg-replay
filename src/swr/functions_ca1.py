import os, mne
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from scipy.io           import savemat
from src.shared.presets import *
from src.shared.utils      import *
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
    

def get_ca1_data(paths, sessions, subjs, get_adj=True, save_mat=False, save_pkl=True):
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
            file = paths.raw_files + subj + '/' + sess + '/' + subj + '_' + sess + paths.suffix
            
            # Check if file exists
            if not os.path.isfile(file):
                print(f'File does not exist: {file}')
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
    if save_pkl: dill_save(lfps, './data/ca1_lfps.pt')

    return lfps