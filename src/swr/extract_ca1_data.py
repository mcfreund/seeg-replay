import pandas as pd
import os
import mne
import matplotlib
from scipy.io import savemat

def find_ca1_contacts(path_imaging):
    '''
    Read CA1 contacts for all participants from imaging excel files.
    '''
    print('\nGetting contact info.')

    # Get all participants in directory from dir names
    participants = os.listdir(path_imaging)

    # Contacts dict with participant as key, timeseries as value
    contacts = dict()
    
    # Cycle over participants
    for participant in participants:
        try:
            # Make excel contact location file name
            imaging_file = path_imaging + participant + '/parsed.xlsx'

            # Notify user and read file
            print(f'Reading: {imaging_file}')
            data = pd.read_excel(imaging_file)

            # Get just the contacts that are in CA1
            contacts[participant] = list(data.loc[data['Location'] == 'CA1']['contact'])    

        # Handle file absence
        except FileNotFoundError:
            print(f"Failure: file not found")
        
        # Handle contact absence
        except KeyError:
            print(f"Failure: File lacks a column for contact Location")
    
    print('Completed.')
    return contacts

def cross_validate_contacts(contacts, data_dict):
    '''
    Checks contacts against data_dict to get list of 
    valid contacts for analysis
    '''

    usable_contacts = {}
    for participant in contacts.keys():
        usable_contacts[participant] = []

        for contact in contacts[participant]:
            if contact.index('-') != -1:
                short_name = contact[:1] + contact[2:]
            else:
                short_name = contact

            if participant in data_dict:
                if type(data_dict[participant][short_name]['ca1_contact']['data']) != list:
                    usable_contacts[participant].append(contact)

    return usable_contacts

def process_contact(df_chinfo, data_raw, contact):
    cdata = dict()
    cdata["data_unfiltered"], cdata["tvec"] = copy_channel_data(data_raw, contact, apply_filt = False)
    cdata["data"], cdata["tvec_filtered"]   = copy_channel_data(data_raw, contact, apply_filt = True)
    cdata["contact_name"] = contact
    cdata["channel_info"] = helper_chinfo(df_chinfo,contact)

    return cdata

def helper_chinfo(df_chinfo, contact_name):
    '''
    Extract appropriate channel info
    '''

    short_name = contact_name[:1] + contact_name[2:] # some contacts lost the hyphen in their name
    if (df_chinfo["contact"] == contact_name).any():
        return df_chinfo[df_chinfo['contact'] == contact_name].iloc[0].to_dict()
    elif (df_chinfo["contact"] == short_name).any():
        return df_chinfo[df_chinfo['contact'] == short_name].iloc[0].to_dict()
    else:
        return ["EMPTY"]

def copy_channel_data(data, contact_name, apply_filt):
    '''
    Extract appropriate full tsd and tvec from 
    appropriate channel
    '''

    # Possible alternate contact name: some contacts lost the hyphen
    alternate_name = contact_name[:1] + contact_name[2:] 

    # Figure out correct field name
    if contact_name in data.ch_names:
        contact_name = contact_name
    elif alternate_name in data.ch_names:
        contact_name = alternate_name
    else:
        return (["EMPTY"], ["EMPTY"])

    # If filtering is requested, do it
    if apply_filt:
        # BP filter range, 80-100 Hz for SWR in humans
        l_freq = 80
        h_freq = 100

        # Get irrelevant channels to drop so we don't filter them needlessly
        irrelevant = [name for name in data.ch_names if name != contact_name]
        data = data.copy()
        data.drop_channels(irrelevant)

        # Filter data
        data.filter(l_freq, h_freq, verbose = 'ERROR')

    # Return data from field
    return data[contact_name]


def get_data(preproc_sfx, path_raw_files, session, participants, contacts, get_adjacent = True):
    '''
    Extracts CA1 contacts and corresponding time series data from each participant's MNE "raw" files.
    
    Inputs:
        preproc_sfx   : Preprocessed data to read. Everything after subject_epoch in _raw.fif filename.
        path_raw_files: Location of subject folders
        session       : Encoding, SameDayRecall, or NextDayRecall
        participants  : List of participant ID strings
        contacts      : Dictionary with subj ID strs as keys, 
        get_adjacent  : Extracts TSD on electrode above and below CA1 on the lead

    Output: Nested dictionary of electrode and 
    '''
    
    print('\nGetting CA1 data from raw files...')
    data_dict = dict()

    # loop through participant
    for participant in participants:
        print(f'Getting data for participant {participant}, session {session}.')

        # Channel dictionary with keys 
        channels = dict()

        # Might not find files, or might not find contacts in list
        try:
            
            # Set current filename
            file = path_raw_files + participant + '/' + session + '/' + participant + '_' + session + preproc_sfx
            
            # Check if file exists
            if not os.path.isfile(file):
                print(f'File does not exist: {file}')
                print('Skipping.')
                continue

            # Read file if it's there
            if file.endswith('.fif'):
                data_raw = mne.io.read_raw_fif(file, preload=True, verbose='ERROR')

            # get info for all channels for this participant
            channel_info_file = f"{path_raw_files}{participant}/{participant}_chinfo.csv"
            print(f'Reading: {channel_info_file}')
            df_chinfo = pd.DataFrame(pd.read_csv(channel_info_file))
            df_chinfo.rename(columns = {'Level 3: gyrus/sulcus/cortex/nucleus': 'Level 3: subregion'}, inplace= True) # shorten for saving to matlab

            # go through channel contacts
            for contact in contacts[participant]:
                print(f'Processing contact: {contact}')

                collect_channel = dict()

                # extract electrode names
                short_name = contact[:1] + contact[2:] # some contacts lost the hyphen in their name
                contact_num = int(contact[-1:])
                contact_below = contact[:-1] + str(contact_num-1)
                contact_above = contact[:-1] + str(contact_num+1)

                # save contact name and time series data
                collect_channel['ca1_contact'] = process_contact(df_chinfo, data_raw, contact)

                # add surrounding electrodes if requested
                if get_adjacent:
                    collect_channel["contact_below"] = process_contact(df_chinfo, data_raw, contact_below)
                    collect_channel["contact_above"] = process_contact(df_chinfo, data_raw, contact_above)

                # 
                channels[short_name] = collect_channel

        # No file
        except FileNotFoundError:
            print(f"Filename error. Check channel_info.csv filename and _raw.fif filenames.")
        
        # No contact
        except KeyError:
            print(f"Participant {participant} lacks one of contacts {contacts[participant]} in data {file}")

        # Save all channels for participant to data
        data_dict[participant] = channels


    return data_dict


# Location of imaging parsed.xlsx files and preprocessed MNE "raw" files
path_task_data = '/oscar/data/brainstorm-ws/seeg_data/Memory Task Data/Imaging/Epilepsy/'
path_raw_files = '/oscar/data/brainstorm-ws/megagroup_data/'
# path_task_data   = '/home/dan/projects/work/seeg_data/original/Imaging/Epilepsy/'
# path_raw_files = '/home/dan/projects/work/megagroup_data/'

# Write .mat file?
save_results = True

# Which files to read, and the session
preproc_sfx = '_no60hz_ref_bp_raw.fif'
#preproc_sfx = 'raw_no60hz_bp_rmouts.fif'
sessions = ['Encoding','SameDayRecall','NextDayRecall']

# Get contact information and participants from imaging files
contacts = find_ca1_contacts(path_task_data)
participants = list(contacts.keys())
#participants = participants[0:1]

# Get data from each session
for sess in sessions:

    # Read raw files and give back CA1 contact time-series
    data_dict = get_data(preproc_sfx, path_raw_files, sess, participants, contacts, get_adjacent = True)

    # cross validate contact information between parsed.xlsx and available time series data
    usable_contacts = cross_validate_contacts(contacts, data_dict)

    # Save in matlab format
    if save_results:
        mdic = {"data": data_dict, "path_raw_files": path_raw_files, "preproc_sfx": preproc_sfx, \
            "contacts": contacts, "usable_contacts": usable_contacts, "participants": participants,
            "session": sess}
        savemat('ca1_data_matrix_' + sess + '.mat', mdic)