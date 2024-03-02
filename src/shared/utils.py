import dill, os, mne
import numpy as np

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
            file = paths.raw_files + subj + '/' + sess + '/' + subj + '_' + sess + paths.suffix

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