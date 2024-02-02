import pandas as pd
import os
import mne
import matplotlib
from scipy.io import savemat

def get_contacts():
    '''
    Get CA1 contacts for all participants
    '''
    mem_path = '/oscar/data/brainstorm-ws/seeg_data/Memory Task Data/'
    img_path = 'Imaging/Epilepsy/'
    participants = os.listdir(os.path.join(mem_path,img_path))
    contacts = dict()
    for participant in participants:
        try:
            imaging_file = mem_path + img_path + participant + '/parsed.xlsx'
            data = pd.read_excel(imaging_file)
            contacts[participant] = list(data.loc[data['Location'] == 'CA1']['contact'])
        except FileNotFoundError:
            print(f"Participant {participant} lacks an imaging file")
        except KeyError:
            print(f"File {imaging_file} lacks a column for contact Location")
    return contacts


def get_channel_info(data_path,folder,participant):
    '''
    Extract info from channel info (produced by src/preproc/clean_raws.py)
    '''
    suffix = '_chinfo.csv'
    df = pd.read_csv(f"{data_path}{participant}/{participant}{suffix}")
    print(df)
    print(df[['contact','was_rereferenced']])
    return df[['contact','was_rereferenced']]


def get_data(file_suffix,data_path,folder,participants):
    '''
    Extracts CA1 contacts per participant 
    '''
    data_dict = dict()
    for participant in participants:
        channels = dict()
        try:
            file = data_path + participant + '/' + folder + '/' + participant + file_suffix
            if file.endswith('.fif'):
                data = mne.io.read_raw_fif(file)
            info = get_channel_info(data_path,folder,participant)
            col = 'was_rereferenced'
            for contact in contacts[participant]:
                contact_misnamed = contact[:1] + contact[2:]
                if contact in data.ch_names:
                    channels[contact_misnamed] = data[contact][0]
                elif contact_misnamed in data.ch_names:
                    channels[contact_misnamed] = data[contact_misnamed][0]
            print(f"\n\nChannels: {channels}\n\n")
            data_dict[participant] = channels
        except FileNotFoundError:
            print(f"Participant {participant} lacks data csv file")
        except KeyError:
            print(f"Participant {participant} lacks one of contacts {contact_list} in data {file}")
    return data_dict

def helper(data, contact_name):
    '''
    Extract appropriate contact from data
    '''
    contact_misnamed = contact_name[:1] + contact_name[2:] # some contacts lost the hyphen in their name
    if contact_name in data.ch_names:
        return data[contact_name][0]
    elif contact_misnamed in data.ch_names:
        return data[contact_misnamed][0]
    else:
        return [0]

def get_data_triplets(file_suffix,data_path,folder,participants):
    '''
    Extracts CA1 contacts per participant and corresponding electrodes
    above and below in depth on the lead

    Saves to dictionary electrode name and 
    '''
    data_dict = dict()
    for participant in participants:
        channels = dict()
        try:
            file = data_path + participant + '/' + folder + '/' + participant + file_suffix
            if file.endswith('.fif'):
                data = mne.io.read_raw_fif(file)
                print(type(data))
            info = get_channel_info(data_path,folder,participant)
            col = 'was_rereferenced'
            for contact in contacts[participant]:

                # extract electrode names
                contact_misnamed = contact[:1] + contact[2:] # some contacts lost the hyphen in their name
                triplet = dict()
                contact_num = int(contact[-1:])
                contact_below = contact[:-1] + str(contact_num-1)
                contact_above = contact[:-1] + str(contact_num+1)

                # save contact name and time series data
                triplet["contact_below"] = dict()
                triplet["contact_below"]["data"] = helper(data, contact_below)
                triplet["contact_below"]["contact_name"] = contact_below

                triplet["ca1_contact"] = dict()
                triplet["ca1_contact"]["data"] = helper(data, contact)
                triplet["ca1_contact"]["contact_name"] = contact

                triplet["contact_above"] = dict()
                triplet["contact_above"]["data"] = helper(data, contact_above)
                triplet["contact_above"]["contact_name"] = contact_above

                channels[contact_misnamed] = triplet
            print(f"\n\nChannels: {triplet}\n\n")
            data_dict[participant] = channels
        except FileNotFoundError:
            print(f"Participant {participant} lacks data csv file")
        except KeyError:
            print(f"Participant {participant} lacks one of contacts {contact_list} in data {file}")
    return data_dict

# file paths
data_path = '/oscar/data/brainstorm-ws/megagroup_data/'
suff = '_Encoding_no60hz_ref_bp_raw.fif'
folder = 'Encoding'

# get contact information and participant list 
contacts = get_contacts()
participants = list(contacts.keys())
participant = participants[0]

# extract time series data
triplet_dict = get_data_triplets(file_suffix=suff,data_path=data_path,folder=folder,participants=participants)

# save in matlab format
mdic = {"data": triplet_dict, "data_path": data_path}
savemat("matlab_matrix.mat", mdic)