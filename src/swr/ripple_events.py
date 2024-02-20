import scipy.io
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
import mne

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



# print(ca1_events[participants[0]][usable_contacts[participants[0]][0]])
# print(usable_contacts[participants[0]])

# behavioral_data = pd.read_csv(path+file3)
# behavioral_data['trial_duration'] = behavioral_data['rt_color'] + behavioral_data['rt_location']

# # There is probably a more efficient way to do this than looping over rows
# # behavioral_data['relevant_start_time'] = 0
# behavioral_data['trial_date_time'] = behavioral_data['trial_date_time'].apply(lambda x: datetime.strptime(x, "%Y,%m,%d,%H,%M,%S.%f"))
# for row in range(behavioral_data.shape[0]):
#     participant = behavioral_data.at[row,'participant_id']
#     session = behavioral_data.at[row,'session']
#     participant_data = behavioral_data.loc[behavioral_data['participant_id'] == participant]
#     session_data = participant_data.loc[participant_data['session'] == session]
#     participant_start_time = session_data.loc[session_data['trial_num']==1,'trial_date_time'].item()
#     # calculate timedelta from datetime of current trial start with the datetime of the first trial start time
#     behavioral_data.at[row,'relevant_start_time'] = behavioral_data.at[row,'trial_date_time'] - participant_start_time

# # convert milliseconds to microseconds (data format to package format) to create timedelta of trial duration
# behavioral_data['trial_duration'] = behavioral_data['trial_duration'].apply(lambda x: timedelta(microseconds=int(x*1000)))
# # add timedelta of trial start time with timedelta of trial duration to create timedelta of trial end time
# behavioral_data['relevant_end_time'] = behavioral_data['relevant_start_time'] + behavioral_data['trial_duration']
# # convert timedelta of trial start and end times into floats as seconds (format of swr data)
# behavioral_data['relevant_start_time'] = behavioral_data['relevant_start_time'].apply(lambda x: x.total_seconds())
# behavioral_data['relevant_end_time'] = behavioral_data['relevant_end_time'].apply(lambda x: x.total_seconds())
# trial_events = {}
# for participant in participants:
#     participant_events = {}
#     participant_data = behavioral_data.loc[behavioral_data['participant_id']==participant]
#     for session in sessions:
#         session_data = participant_data.loc[participant_data['session']==session]
#         session_events = session_data[['relevant_start_time','relevant_end_time']].values
#         participant_events[session] = session_events
#     trial_events[participant] = participant_events


def itiHelper(swr_array,trial_array):
    swr_reshaped = swr_array[:, np.newaxis, np.newaxis]
    trial_reshaped = trial_array[np.newaxis, :, :]
    # comparison array tells you which behavioral event markers the SWR occurred after
    comparison_array = swr_reshaped > trial_reshaped
    #print(comparison_array.shape)
    # location array tells you how many behavioral event markers (out of start and end) the SWR occurred after for each trial
    location_array = comparison_array.sum(axis=2)
    #print(location_array.sum(axis=1) == (trial_array.shape[0] * 2))
    # intersection array finds which SWRs occurred during which trials (i.e. they have seen more starts than ends)
    intersection_array = location_array%2
    # print(intersection_array)
    #print((location_array == (trial_array.shape[0] * 2)).sum())
    #print(iti_intersection_array)
    # swr intersections array sums over trials to find which SWRs occurred during a trial
    swr_intersections = intersection_array.sum(axis=1)
    #print(swr_intersections)
    # iti intersection array sends all SWRs that occurred after the experiment to -1
    iti_swr_intersections = swr_intersections - (location_array.sum(axis=1) == (trial_array.shape[0] * 2))
    #print(iti_swr_intersections)
    # iti_swrs hence indicates which SWRs did NOT occur during a trial
    trial_swrs = ((iti_swr_intersections == 1).sum())
    iti_swrs = (iti_swr_intersections == 0).sum()
    ps_swrs = (iti_swr_intersections == -1).sum()
    total_swrs = intersection_array.shape[0]
    trial_duration_total = (trial_array[:,1] - trial_array[:,0]).sum()
    session_duration = trial_array[-1,1]
    iti_duration_total = session_duration - trial_duration_total
    recording_duration = swr_array[-1]
    post_session_duration = recording_duration - session_duration
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
    #iti_swr_ids = np.where(iti_swrs==1)
    #print(f'ITI SWR ids: {iti_swr_ids}')



def isinITI(participants, sessions, swr_events, trial_events):
    for participant in participants:
        # swr events are not currently organized by session - only participant
        # current available data only checks Encoding
        participant_contacts = swr_events[participant]
        participant_sessions = trial_events[participant]
        for session in sessions:
            if session=='Encoding':
                for contact in participant_contacts:
                    participant_swrs = participant_contacts[contact]
                    participant_trials = participant_sessions[session]
                    # SWRs are assumed to be so fast that we only need to check the start time...
                    # ...when comparing them to behavioral data
                    print(f'\n-----\nParticipant: {participant}\nSession: {session}\nContact: {contact}\n-----\n')
                    itiHelper(participant_swrs[:,0], participant_trials)
            else:
                raise NotImplementedError


def get_raw_data(file_suffix,data_path,folder,participant):
    file = data_path + participant + '/' + folder + '/' + participant + file_suffix
    raw = mne.io.read_raw_fif(file, preload=True)
    events = mne.events_from_annotations(raw)
    return events

data_path = '/oscar/data/brainstorm-ws/megagroup_data/'
file_suffix = '_Encoding_no60hz_ref_bp_raw.fif'    # preprocessed target
sessions = ['Encoding']                         # session
trial_events = {}
for participant in participants:
    participant_events = {}
    for session in sessions:
        array, dic = get_raw_data(file_suffix=file_suffix,data_path=data_path, folder=session,participant=participant)
        trial_starts = np.where(array==dic['loc_start'])[0]
        trial_stops = np.where(array==dic['col_resp'])[0]
        participant_events[session] = np.stack((trial_starts,trial_stops),axis=-1)
    trial_events[participant] = participant_events


isinITI(participants=participants, sessions=sessions, swr_events=ca1_events, trial_events=trial_events)



# print(type(mat_data['data']))
# print(mat_data['data'].shape)
# print(mat_data['data'].dtype.names)
# print(type(mat_data['data'][participant_0]))
# print(mat_data['data'][participant_0].shape)
# print(mat_data['data'][participant_0][0][0]['CMHIP2'][0][0]['ca1_contact'][0][0]['tstart'][0][0])