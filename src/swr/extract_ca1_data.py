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

def cross_validate_contacts(contacts, data_dict):
    '''
    Checks contacts against data_dict to get list of 
    valid contacts for analysis
    '''
    usable_contacts = {}
    for participant in contacts.keys():
        contact_sublist = []
        data_channels = data_dict[participant]
        for contact in contacts[participant]:
            if contact.index('-') != -1:
                contact_misnamed = contact[:1] + contact[2:]
            else:
                contact_misnamed = contact
            if type(data_channels[contact_misnamed]['ca1_contact']['data']) != list:
                contact_sublist.append(contact)
        usable_contacts[participant] = contact_sublist
    return usable_contacts


def helper_chinfo(df_chinfo, contact_name):
    '''
    Extract appropriate channel info
    '''

    contact_misnamed = contact_name[:1] + contact_name[2:] # some contacts lost the hyphen in their name
    if (df_chinfo["contact"] == contact_name).any():
        return df_chinfo[df_chinfo['contact'] == contact_name].iloc[0].to_dict()
    elif (df_chinfo["contact"] == contact_misnamed).any():
        return df_chinfo[df_chinfo['contact'] == contact_misnamed].iloc[0].to_dict()
    else:
        return ["EMPTY"]

def helper_data(data, contact_name):
    '''
    Extract appropriate full tsd and tvec from 
    appropriate channel
    '''
    contact_misnamed = contact_name[:1] + contact_name[2:] # some contacts lost the hyphen in their name
    if contact_name in data.ch_names:
        return data[contact_name]
    elif contact_misnamed in data.ch_names:
        return data[contact_misnamed]
    else:
        return (["EMPTY"], ["EMPTY"])

def get_data(file_suffix,data_path,folder,participants,contacts,triplet=True):
    '''
    Extracts CA1 contacts per participant and corresponding time series data.
    Triplet = True, extracts TSD on electrode above and below CA1 on the lead

    Output: Nested dictionary of electrode and 
    '''
    data_dict = dict()

    # loop through participant
    for participant in participants:

        channels = dict()

        try:
            # get tsd file 
            file = data_path + participant + '/' + folder + '/' + participant + file_suffix
            if file.endswith('.fif'):
                data_raw = mne.io.read_raw_fif(file)

            # get info for all channels for this participant
            df_chinfo = pd.DataFrame(pd.read_csv(f"{data_path}{participant}/{participant}_chinfo.csv"))
            df_chinfo.rename(columns = {'Level 3: gyrus/sulcus/cortex/nucleus': 'Level 3: subregion'}, inplace= True) # shorten for saving to matlab

            # go through channel contacts
            for contact in contacts[participant]:
                
                collect_channel = dict()

                # extract electrode names
                contact_misnamed = contact[:1] + contact[2:] # some contacts lost the hyphen in their name
                contact_num = int(contact[-1:])
                contact_below = contact[:-1] + str(contact_num-1)
                contact_above = contact[:-1] + str(contact_num+1)

                # save contact name and time series data
                collect_channel["ca1_contact"] = dict()
                collect_channel["ca1_contact"]["data"], collect_channel["ca1_contact"]["tvec"]  = helper_data(data_raw, contact)
                collect_channel["ca1_contact"]["contact_name"] = contact
                collect_channel["ca1_contact"]["channel_info"] = helper_chinfo(df_chinfo,contact)


                # add surrounding electrodes if requested
                if triplet:
                    collect_channel["contact_below"] = dict()
                    collect_channel["contact_below"]["data"], collect_channel["contact_below"]["tvec"] = helper_data(data_raw, contact_below)
                    collect_channel["contact_below"]["contact_name"] = contact_below
                    collect_channel["contact_below"]["channel_info"] = helper_chinfo(df_chinfo,contact_below)

                    collect_channel["contact_above"] = dict()
                    collect_channel["contact_above"]["data"], collect_channel["contact_above"]["tvec"] = helper_data(data_raw, contact_above)
                    collect_channel["contact_above"]["contact_name"] = contact_above
                    collect_channel["contact_above"]["channel_info"] = helper_chinfo(df_chinfo,contact_above)

                channels[contact_misnamed] = collect_channel

            # save all channels for participant to data
            data_dict[participant] = channels

        except FileNotFoundError:
            print(f"Participant {participant} lacks data csv file")
        except KeyError:
            print(f"Participant {participant} lacks one of contacts {contact_list} in data {file}")
    return data_dict

# desired file paths
data_path = '/oscar/data/brainstorm-ws/megagroup_data/'
suff = '_Encoding_no60hz_ref_bp_raw.fif'    # preprocessed target
folder = 'Encoding'                         # session

# get contact information and participant list 
contacts = get_contacts()
participants = list(contacts.keys())
participant = participants[0]

# extract time series data
data_dict = get_data(file_suffix=suff,data_path=data_path, \
    folder=folder,participants=participants,contacts=contacts, \
        triplet = True)

# cross validate contact information between parsed.xlsx and available time series data
usable_contacts = cross_validate_contacts(contacts, data_dict)

# save in matlab format
mdic = {"data": data_dict, "data_path": data_path, "data_file": suff, \
    "contacts": contacts, "usable_contacts": usable_contacts, "participants": participants}
savemat("ca1_data_matrix.mat", mdic)