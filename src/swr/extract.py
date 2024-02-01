import pandas as pd
import os
import mne

def get_contacts():
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
    suffix = '_chinfo.csv'
    df = pd.read_csv(f"{data_path}{participant}/{participant}{suffix}")
    return df[['contact','was_rereferenced']]


def get_data(file_suffix,data_path,folder,participants):
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
                    channels[contact] = data[contact]
                elif contact_misnamed in data.ch_names:
                    channels[contact] = data[contact_misnamed]
            print(f"\n\nChannels: {channels}\n\n")
            data_dict[participant] = channels
        except FileNotFoundError:
            print(f"Participant {participant} lacks data csv file")
        except KeyError:
            print(f"Participant {participant} lacks one of contacts {contact_list} in data {file}")
    return data_dict


data_path = '/oscar/data/brainstorm-ws/megagroup_data/'
suff = '_Encoding_no60hz_ref_bp_raw.fif'
folder = 'Encoding'
contacts = get_contacts()
participants = list(contacts.keys())
participant = participants[0]
info = pd.read_csv(f"{data_path}{participant}/{participant}_chinfo.csv")
data_dict = get_data(file_suffix=suff,data_path=data_path,folder=folder,participants=participants)
print(contacts)
print(data_dict)

