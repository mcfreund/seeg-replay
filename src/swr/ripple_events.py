import scipy.io
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
import mne

file_suffix = '_Encoding_no60hz_ref_bp_raw.fif'
raw_path = "/oscar/data/brainstorm-ws/megagroup_data/"
matlab_path = "seeg-replay/src/swr/"
matlab_pre = "ca1_data_matrix.mat"
matlab_post = "event_ca1_data.mat"
participants = ['e0010GP', 'e0011XQ', 'e0013LW', 'e0015TJ', 'e0016YR', 'e0017MC']
sessions = ['Encoding',
#'SameDayRecall','NextDayRecall'
]

# loading in data
mat_data_pre = scipy.io.loadmat(matlab_path+matlab_pre)
mat_data = scipy.io.loadmat(matlab_path+matlab_post)


### Functions

# get metadata for trial start / stop times
def get_trials():
    trial_events = {}
    # iterate over participants
    for participant in participants:
        participant_events = {}
        # iterate over sessions
        for session in sessions:
            # load raw data
            file = raw_path + participant + '/' + session + '/' + participant + file_suffix
            raw = mne.io.read_raw_fif(file, preload=True)
            # get metadata for trial start / stop times
            array, dic = mne.events_from_annotations(raw)
            trial_starts = np.where(array==dic['trial_start'])[0]
            trial_stops = np.where(array==dic['trial_stop'])[0]
            # save metadata in nested dictionary
            participant_events[session] = np.stack((trial_starts,trial_stops),axis=-1)
        trial_events[participant] = participant_events
    return trial_events

# helper functino for finding ITIs and calculating and print relevant information
def itiHelper(swr_array,trial_array):
    swr_reshaped = swr_array[:, np.newaxis, np.newaxis]
    trial_reshaped = trial_array[np.newaxis, :, :]
    # comparison array tells you which behavioral event markers the SWR occurred after
    comparison_array = swr_reshaped > trial_reshaped
    # location array tells you how many behavioral event markers (out of start and end) the SWR occurred after for each trial
    location_array = comparison_array.sum(axis=2)
    # intersection array finds which SWRs occurred during which trials (i.e. they have seen more starts than ends)
    intersection_array = location_array%2
    # swr intersections array sums over trials to find which SWRs occurred during a trial
    swr_intersections = intersection_array.sum(axis=1)
    # iti intersection array sends all SWRs that occurred after the experiment to -1
    iti_swr_intersections = swr_intersections - (location_array.sum(axis=1) == (trial_array.shape[0] * 2))
    # iti_swrs hence indicates which SWRs did NOT occur during a trial
    # get counts for swr subcategories
    trial_swrs = ((iti_swr_intersections == 1).sum())
    iti_swrs = (iti_swr_intersections == 0).sum()
    ps_swrs = (iti_swr_intersections == -1).sum()
    total_swrs = intersection_array.shape[0]
    # calculating durations
    trial_duration_total = (trial_array[:,1] - trial_array[:,0]).sum()
    session_duration = trial_array[-1,1]
    iti_duration_total = session_duration - trial_duration_total
    recording_duration = swr_array[-1]
    post_session_duration = recording_duration - session_duration
    # printing off information
    print(f'Total ITI SWRs: {iti_swrs} out of {total_swrs-ps_swrs} SWRs within the experiment')
    print(f'Meaning {iti_swrs/(total_swrs-ps_swrs)*100:.1f}% of experiment SWRs were ITI SWRs')
    print(f'ITI time made up {(iti_duration_total)/session_duration*100:.1f}% of the experiment time')
    print(f'Total ITI SWRs: {iti_swrs} out of {total_swrs} total SWRs')
    print(f'Meaning {iti_swrs/total_swrs*100:.1f}% of SWRs were ITI SWRs')
    print(f'ITI time made up {(iti_duration_total)/recording_duration*100:.1f}% of the SEEG recording time')
    print(f'Total experiment SWRs: {total_swrs-ps_swrs} out of {total_swrs} total SWRs')
    print(f'Meaning {(total_swrs-ps_swrs)/total_swrs*100:.1f}% of SWRs were experiment SWRs')
    print(f'The experiment made up {(session_duration)/recording_duration*100:.1f}% of the SEEG recording time')
    print(f'SWRs occurred {(ps_swrs/post_session_duration)/((total_swrs-ps_swrs)/session_duration):.1f} times more frequently after the experiment than during')
    print(f'SWRs occurred {(iti_swrs/iti_duration_total)/((trial_swrs)/(trial_duration_total)):.1f} times more frequently in between tasks than during them')

trials = get_trials()

# loop
for participant in participants:
    # get contacts
    contacts = mat_data_pre['usable_contacts'][participant][0][0]
    participant_trials = trials[participant]
    # swr events are not currently organized by session - only participant
    # current available data only checks Encoding
    for session in sessions:
        session_trials = participant_trials[session]
        if session=='Encoding':
            for contact in contacts:
                # matlab removes hyphen in name
                contact_misnamed = contact[:1] + contact[2:]
                # extract start and end times of SWRs
                tstart = mat_data['data'][participant][0][0][contact_misnamed][0][0]['ca1_contact'][0][0]['tstart'][0][0]
                tend = mat_data['data'][participant][0][0][contact_misnamed][0][0]['ca1_contact'][0][0]['tend'][0][0]
                # convert to numpy
                ca1_events = np.stack([tstart,tend],axis=1)[:,:,0]
                print(f'\n-----\nParticipant: {participant}\nSession: {session}\nContact: {contact}\n-----\n')
                # we only check when SWRs start
                start_times = ca1_events[:,0]
                itiHelper(start_times, session_trials)
        else:
            raise NotImplementedError