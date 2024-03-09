import mne
import os
import pandas as pd
import scipy.io
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
import matplotlib.pyplot as plt

matlab_path = 'seeg-replay/src/swr/'
raw_path = '/oscar/data/brainstorm-ws/megagroup_data/'
file_suffix = '_Encoding_no60hz_ref_bp_raw.fif'
matlab_pre = "ca1_data_matrix.mat"
matlab_post = "event_ca1_data.mat"
file3 = "behavioral_data.csv"
participants = ['e0010GP', 'e0011XQ', 'e0013LW', 'e0015TJ', 'e0016YR', 'e0017MC']
sessions = ['Encoding',
#'SameDayRecall','NextDayRecall'
]

mat_data_pre = scipy.io.loadmat(matlab_path+matlab_pre)
mat_data = scipy.io.loadmat(matlab_path+matlab_post)

# iterate over sessions
for session in sessions:
    # iterate over participants
    for participant in participants:
        # read in raw mne data
        file = raw_path + participant + '/' + session + '/' + participant + file_suffix
        raw = mne.io.read_raw_fif(file, preload=True)
        events = mne.events_from_annotations(raw)
        array, dic = events
        # add SWR events at each contact to annotations
        contacts = mat_data_pre['usable_contacts'][participant][0][0]
        for contact in contacts:
            # matlab removes hyphen in name
            contact_misnamed = contact[:1] + contact[2:]
            # extract start and end times of SWRs
            tstart = mat_data['data'][participant][0][0][contact_misnamed][0][0]['ca1_contact'][0][0]['tstart'][0][0]
            tend = mat_data['data'][participant][0][0][contact_misnamed][0][0]['ca1_contact'][0][0]['tend'][0][0]
            # convert to numpy
            ca1_events = np.stack([tstart,tend],axis=1)[:,:,0]
            # add SWR annotations to ca1 raw channel data
            raw.set_annotations(mne.Annotations(
                onset=ca1_events[:,0],
                duration=ca1_events[:,1]-ca1_events[:,0],
                description=[f"SWR:{contact}" for _ in ca1_events[:,0]],
            )+raw.annotations)
        # remove bad_outlier annotations
        annot_df = raw.annotations.to_data_frame()
        idx_delete = annot_df.index[annot_df['description'] == "BAD_outlier"].tolist()
        raw.annotations.delete(idx_delete)
        # plot ca1 channel(s) with annotations
        if len(contacts) > 0:
            try:
                raw.pick(contacts).plot(n_channels=len(contacts), show=True, block=True)
            except ValueError:
                raw.pick([name[:1]+name[2:] for name in contacts]).plot(n_channels=len(contacts), show=True, block=True)
        raw.plot()
        plt.show()

print(events)
