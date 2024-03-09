import dill, os, mne
import numpy  as np
import pandas as pd

def dill_save(obj, fname):
    with open(fname,'wb') as file:
        dill.dump(obj, file)

def dill_read(fname):
    with open(fname,'rb') as file:
        obj = dill.load(file)
    return obj

def get_trial_times_(paths, subjs, sessions):
    ''' Extracts start and stop times for every trial from MNE raw.fif file.'''

    # Trial times data structure. The ususal dictionary with subject and session
    trial_times = {subj:{sess:[] for sess in sessions} for subj in subjs}

    for subj in subjs:
        for sess in sessions:

            # Generate filename
            file = paths.processed_raws + subj + '/' + sess + '/' + subj + '_' + sess + paths.suffix

            # Check if file is there, then read or skip
            if os.path.isfile(file):
                print(f'Reading file: {file}')
                raw = mne.io.read_raw_fif(file, preload=True, verbose='ERROR')
            else:
                print(f'No such file: {file}')
                continue

            # Get all annotated event tuples and dictionary of annotations
            array, dic = mne.events_from_annotations(raw, verbose='Error')

            # Find only the start and stop indices
            trial_beg_inds = np.where(array==dic['trial_start'])[0]
            trial_end_inds = np.where(array==dic['trial_stop'])[0]

            # Convert to times
            trial_beg = array[trial_beg_inds,0]
            trial_end = array[trial_end_inds,0]

            #
            trial_times[subj][sess] = np.stack((trial_beg,trial_end),axis=-1)

    # Save start and end times
    return trial_times


def find_contacts(paths, subjs, loc='CA1'):
    '''
    Read contact names for a location from imaging excel files.

    Args:
        paths: Must include paths.imaging (see paths)
        subjs: List of subjects to get information for.
        loc  : Location field of parsed.xlsx to match. '*' gets all.
    '''
    print('\nGetting contact info.')

    # Get all subjs in directory from dir names
    #subjs = os.listdir(paths.imaging)

    # Contacts dict with subj as key, timeseries as value
    contacts = dict()
    
    # Cycle over subjs
    for subj in subjs:
        try:
            # Make excel contact location file name
            imaging_file = paths.imaging + subj + '/parsed.xlsx'

            # Notify user and read file
            print(f'Reading: {imaging_file}')
            data = pd.read_excel(imaging_file)

            # Get just the contacts that are in requested location
            if loc == '*':
                contacts[subj] = list(data['contact'])
            else:
                contacts[subj] = list(data.loc[data['Location'] == loc]['contact'])                

        # Handle file absence
        except FileNotFoundError:
            print(f"Failure: file not found")
        
        # Handle contact absence
        except KeyError:
            print(f"Failure: File lacks a column for contact Location")
    
    print('Completed.')
    return contacts


def check_contact_exists(contact_name, data_raw):
    '''Checks a raw file for a contact field.
    
    Args:
        contact_name: Name in parsed.xlsx file such as A-AMY4
        data_raw    : mne raw object to look for name in.
    
    Returns:
        (truth value, name string)
    '''
    # Possible alternate contact name: some contacts lost the hyphen
    alternate = contact_name[:1] + contact_name[2:] 

    # Figure out correct field name
    if contact_name in data_raw.ch_names:
        return True, contact_name
    elif alternate in data_raw.ch_names:
        return True, alternate
    else:
        return False, None
    

def get_adjacent_contacts(contact):

    contact_num = int(contact[-1:])
    contact_below = contact[:-1] + str(contact_num-1)
    contact_above = contact[:-1] + str(contact_num+1)

    #above_exists, contact_above = check_contact_exists(contact_above, raw)
    #below_exists, contact_below = check_contact_exists(contact_below, raw)

    #if ~above_exists: contact_above = None
    #if ~below_exists: contact_below = None

    return contact_above, contact_below