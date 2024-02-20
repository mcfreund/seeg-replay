import mne
import os
import pandas as pd
import scipy.io
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
import matplotlib.pyplot as plt


path = "/oscar/data/brainstorm-ws/megagroup_data/"
file = "event_ca1_data.mat"
file2 = "ca1_data_matrix.mat"
file3 = "behavioral_data.csv"
participants = ['e0010GP', 'e0011XQ', 'e0013LW', 'e0015TJ', 'e0016YR', 'e0017MC']
sessions = ['Encoding',
#'SameDayRecall','NextDayRecall'
]
participant_0 = participants[0]

mat_data = scipy.io.loadmat(path+file)
mat_data_pre = scipy.io.loadmat(file2)

usable_contacts = dict()
for participant in participants:
    usable_contact_list = mat_data_pre['usable_contacts'][participant][0][0]
    if len(usable_contact_list) > 0:
        usable_contacts[participant] = list(usable_contact_list)
    else:
        usable_contacts[participant] = []

print(usable_contacts)

ca1_events = dict()

for participant in participants:
    ca1_events[participant] = dict()
    data = mat_data['data'][participant][0][0]
    for contact in usable_contacts[participant]:
        contact_misnamed = contact[:1] + contact[2:]
        tstart = data[contact_misnamed][0][0]['ca1_contact'][0][0]['tstart'][0][0]
        tend = data[contact_misnamed][0][0]['ca1_contact'][0][0]['tend'][0][0]
        ca1_events[participant][contact] = np.stack([tstart,tend],axis=1)[:,:,0]

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

data_path = '/oscar/data/brainstorm-ws/megagroup_data/'
file_suffix = '_Encoding_no60hz_ref_bp_raw.fif'    # preprocessed target
folder = 'Encoding'                         # session
contacts = get_contacts()
participants = list(contacts.keys())

for participant in participants:
    file = data_path + participant + '/' + folder + '/' + participant + file_suffix
    raw = mne.io.read_raw_fif(file, preload=True)
    events = mne.events_from_annotations(raw)
    array, dic = events
    trial_starts = np.where(array==dic['trial_start'])[0]
    trial_stops = np.where(array==dic['trial_stop'])[0]
    for contact in usable_contacts[participant]:
        raw.set_annotations(mne.Annotations(
            onset=ca1_events[participant][contact][:,0],
            duration=ca1_events[participant][contact][:,1]-ca1_events[participant][contact][:,0],
            description=[f"SWR:{contact}" for _ in ca1_events[participant][contact][:,0]],
        )+raw.annotations)
    contacts = usable_contacts[participant]
    if len(contacts) > 0:
        try:
            raw.pick(contacts).plot(n_channels=len(contacts), show=True, block=True)
        except ValueError:
            raw.pick([name[:1]+name[2:] for name in contacts]).plot(n_channels=len(contacts), show=True, block=True)
    raw.plot()
    plt.show()
        
#epoch_ends = np.where(array==dic['clip_start'])[0]

print(array[trial_starts, 0])
print(array[trial_stops, 0])
print(events)
