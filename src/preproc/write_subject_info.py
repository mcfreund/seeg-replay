## most of this script is adapted from: https://github.com/Brainstorm-Program/brainstorm_challenge_2024/blob/main/scripts/brainstorm_reorganize_data.py
## it was modified to write a single csv file, `megagroup_dat/session_info.csv`, which contains info about each acquisition session, including the condition
## (encoding, same-day recall, next-day recall) and the corresponding location of neural data files.
## this csv is read by subsequent preprocessing/analysis scripts.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import itertools
import pickle
from tqdm import tqdm
import bz2
import _pickle as cPickle

dir_output = '/oscar/data/brainstorm-ws/megagroup_data/'


def load_mat(fmat):
    return sio.loadmat(fmat, struct_as_record=False, squeeze_me=True)

def mat_to_dict(mat):
    """ convert mat_struct object to a dictionary recursively
    """
    dict = {}

    for field in mat._fieldnames:
        val = mat.__dict__[field]
        if isinstance(val, sio.matlab.mat_struct):
            dict[field] = mat_to_dict(val)
        else:
            dict[field] = val

    return dict

def read_bhv(behavior):
    """ load behavioral data from .mat file and extra trial data
        returns dictionary of all behavioral data per trial
    """
    bhv = {}
    existing_trials = 0

    f = load_mat(behavior)
    for k in f.keys():
        if k[0:5] == 'Trial' and k[5:].isdigit():
            trial = f[k]
            trial_num = int(k[5:]) + existing_trials
            bhv[trial_num] = mat_to_dict(trial)

    return bhv

def load_pkl(path):
    """ loads compressed pickle file called by load_electrodes() """

    with bz2.open(path, 'rb') as f:
        neural_data = cPickle.load(f)
    return neural_data

def find_trials(events, verbose=False):
    """ finds *all* trials with associated codes in an events file """
    events_signal = load_pkl(events)
    codes = events_signal['signal']
    fs = events_signal['fs']

    trials = {0: []}

    count = 0
    for c in codes:
        if c[0] == 9:
            count += 1
            trials[count] = [c]
            if count > 1 and trials[count-1][-1][0] != 18 and verbose:
                # assert trials[count-1][-1][0] == 18
                print('WARNING: parsed trial {} does not end in 18'.format(count))
        else:
            trials[count].append(c)

    if verbose:
        print('found {} trials with {} codes'.format(count, len(codes)))

    return fs, codes, trials

def ranges(i):
    """ just provides better print() output for trials """
    for a, b in itertools.groupby(enumerate(i), lambda pair: pair[1] - pair[0]):
        b = list(b)
        yield b[0][1], b[-1][1]

# this is the work-horse function that aligns the trials in the behavioral files with those in the neural ones
def match_trials(bhv, codes, trials, fs, margin=0.1, verbose=False):
    """ matches codes from behavioral data with neural events - returns dictionary of trials """
    matches = {}
    c = 0
    cmax = 0
    for tr in bhv:
        bhv_tr_len = np.asarray(bhv[tr]['BehavioralCodes']['CodeTimes'][-1] - bhv[tr]['BehavioralCodes']['CodeTimes'][0], dtype=int)
        m = False
        while c < len(trials) and m == False:
            # check for valid trials
            if len(trials[c]) > 1:
                tr_codes = [j[0] for j in trials[c]]
                if trials[c][0][0] == 9 and 18 in tr_codes:
                    tr_start = trials[c][0][1]

                    i18 = tr_codes.index(18)

                    tr_end = trials[c][i18][1]
                    nrl_tr_len = (tr_end - tr_start) / fs * 1000 # convert to milliseconds
                    diff = 1 - nrl_tr_len/bhv_tr_len
                    if abs(diff) < margin:
                        if verbose:
                            print('Aligned Trial {} (acc: {:.3f}%)'.format(tr, diff*100))
                        matches[tr] = trials[c]
                        m = True
                        c += 1
                        cmax = c
                    else:
                        c += 1
                else:
                    c += 1
            else:
                c+=1
        c = cmax

        if m == False and verbose:
            print('could not find match for {}'.format(tr))

    match_range = ranges(matches.keys())

    rstr = []
    for i in match_range:
        if i[0] == i[1]:
            rstr.append('{}'.format(i[0]))
        else:
            rstr.append('{}-{}'.format(i[0], i[1]))

    match_range = ', '.join(list(rstr))
    print('found matches for trials {}'.format(match_range))

    return matches



## to become cols of pandas data.frame:
subject_, day_, date_, phase_, day_id, subdir_orig = [], [], [], [], [], []
matched_trials_all = []  ## to be written as separate csv.

data_path = os.path.abspath('/oscar/data/brainstorm-ws/seeg_data/Memory Task Data/Epilepsy/Monitoring/')

#Manually define participant IDs
participants_of_interest = [
    ['e0010GP_00', 'e0010GP_01'],
    ['e0011XQ_00', 'e0011XQ_01'],
    ['e0017MC_00', 'e0017MC_01'],
    #['e0019VQ_00', 'e0019VQ_01'],  ## no anatomical data
    #['e0020JA_00', 'e0020JA_01'],
    #['e0024DV_00', 'e0024DV_01'],
    ['e0013LW_02', 'e0013LW_03'], 
    ['e0015TJ_01', 'e0015TJ_02'], 
    #['e0014VG_00', 'e0014VG_00'], #No next day recall, so reuse first day filename temporarily
    ['e0016YR_02', 'e0016YR_02'], #No next day recall, so reuse first day filename temporarily
    #['e0022ZG_00', 'e0022ZG_00']  #No next day recall, so reuse first day filename temporarily
]

#Determine filenames 
participants_of_interest_flattened = [participant_filename for participant_filenames in participants_of_interest for participant_filename in participant_filenames] #Determines the actual filenames (including the date) for each file
participant_fullnames_flattened = [directory_file for participant_file in participants_of_interest_flattened for directory_file in os.listdir(data_path) if participant_file in directory_file] #Flattens the list to be 1-dimensional
participant_fullnames = np.array(participant_fullnames_flattened).reshape(-1,2).tolist() #Reformats to a 2D array
participant_fullnames = [file if file[0] != file[1] else [file[0]] for row, file in enumerate(participant_fullnames)] #We will remove second day filenames for participants that did not have a second day

#Iterates through all participants
for participant_filenames in participant_fullnames: 
    
    #Create participant specific dataframes
    times = [f"Time{str(datapoint).zfill(4)}" for datapoint in range(5120)]
    participant_data = pd.DataFrame(columns=['Participant_ID','Phase','Condition','Electrode','Error_Position','Error_Color']+times)
    
    #Iterates through Day 0 and Day 1. 
    #Day 0 includes the encoding phase and the same-day recall phase while Day 1 includes the next-day recall phase.
    for day, participant_filename in enumerate(participant_filenames):
        print(f"\nParticipant Filename: {participant_filename}")

        #Determine metadata and filenames
        participant_id = participant_filename.split('_')[-2]
        participants_filenames = os.listdir(f"{data_path}/{participant_filename}")
        participant_electrodes = [p_file.split('-')[-1].replace('.pbz2','') for p_file in participants_filenames if '.pbz2' in p_file and 'Events.pbz2' not in p_file]
        participant_neural_filenames = [p_file for p_file in participants_filenames if '.pbz2' in p_file and 'Events.pbz2' not in p_file]

        #Iterate through phases.
        #Day 0 contains Phases A and B, which are encoding and same-day recall, respectively
        #Day 1 contains Phase A (which is different from Day 0's phase A), which is next-day recall.
        phases = [['A','B'] if day == 0 else ['A']][0]
        phases = ['C'] if day == 1 and participant_id == 'e0015TJ' else phases #One participant has a different naming convention
        
        #Iterate through the phases we just defined
        for phase in phases:
            
            #Determine filenames
            participant_beh = f"{data_path}/{participant_filename}/{participant_filename[:-2]}{phase}.mat"
            participant_events = f"{data_path}/{participant_filename}/{participant_filename[:-2]}Events.pbz2"
            
            #Create a label to better signify the current phase
            if day == 0 and phase == 'A':
                phase_dict = 'Encoding'
            elif day == 0 and phase == 'B':
                phase_dict = 'SameDayRecall'
            elif day == 1:
                phase_dict = 'NextDayRecall'

            #Print report
            print(f"\nID: {participant_id}")
            print(f"Day: {day}")
            print(f"Phase: {phase_dict}")

            #Load and navigate the data using Bryan Zheng's functions
            bhv = read_bhv(participant_beh)
            fs, codes, trials = find_trials(participant_events)
            matched_trials = match_trials(bhv, codes, trials, fs)
           
            ## save info:
            subject_.append(participant_id)
            day_id.append(participant_filename.split('_')[-1])
            date_.append(participant_filename.split('_')[0])
            subdir_orig.append(participant_filename)
            phase_.append(phase_dict)
            matched_trials_all.append(matched_trials)

## concatenate into a data.frame
df = pd.DataFrame(
    {'participant_id': subject_,
     'day_id': day_id,
     'subdir_orig': subdir_orig,
     'date': date_,
     'session': phase_
    })

subdirnames = []
for (i, d), matched_trials in zip(df.iterrows(), matched_trials_all):
    ## create subject directory:
    subdirname = os.path.join(d["participant_id"], d["session"])
    subdirnames.append(subdirname)
    dir_sess = os.path.join(dir_output, subdirname)
    os.makedirs(dir_sess, exist_ok = True)
    ## write events:
    events = np.concatenate([trials for trials in matched_trials.values()])
    fn = os.path.join(dir_sess, d["participant_id"] + "_" + d["session"] + "_events.csv")
    pd.DataFrame(events).rename(columns={0: 'code', 1: 'time'}).to_csv(fn, index = False)

df["subdir_data"] = subdirnames
df.to_csv(os.path.join(dir_output, "session_info.csv"), index = False)
