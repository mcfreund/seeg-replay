'''
This file contains the main preprocessing functions.
'''
# Shared imports
import os
import mne
import numpy  as np
import pandas as pd

from src.preproc.utils import save_raw_if, save_plt_if

# This function does several things which should be disaggregated:
# - read behavioral data and write an events.csv file
# - write a session_info.csv file tracking the file structure

def make_session_info(params):
    '''
    This function writes a session_info.csv file containing session metadata, which is used in preprocessing.

    It is adapted to some extent from:
    https://github.com/Brainstorm-Program/brainstorm_challenge_2024/blob/main/scripts/brainstorm_reorganize_data.py
    '''

    from src.preproc.utils import read_bhv, find_trials, match_trials

    # Data frame for session info
    df = pd.DataFrame(columns = ['participant_id', 'day_id', 'subdir_orig', 'date', 'session'])
    matched_trials_all = []

    # List of subject directories
    subj_dirs = [
        '2020-11-12_e0010GP_00',
        '2020-11-13_e0010GP_01',
        '2021-01-17_e0011XQ_00',
        '2021-01-18_e0011XQ_01',
        '2023-03-01_e0017MC_00',
        '2023-03-02_e0017MC_01',
        '2021-10-01_e0013LW_02',
        '2021-10-02_e0013LW_03',
        '2022-11-10_e0015TJ_01',
        '2022-11-11_e0015TJ_02',
        '2023-02-07_e0016YR_02']

    # Corrosponding days
    subj_days = [0,1, 0,1, 0,1, 0,1, 0,1, 0]

    # For each participant session, collect session info into frame
    for day, dir in zip(subj_days, subj_dirs):

        # Display which subject is being processed
        print(f"\nParticipant Day, Directory: {day}, {dir}")

        # Get subject ID and all files in directory
        subj_id = dir.split('_')[-2]
        subj_fnames = os.listdir(f"{params.path_read}/{dir}")
        
        # Get electrode list and data list
        subj_electrodes    = []
        subj_neural_fnames = []
        
        # Electrodes are from filenames
        for file in subj_fnames:

            # All .pbz2 files except events file are electrode files
            if '.pbz2' in file and 'Events.pbz2' not in file:
                electrode = file.split('-')[-1].replace('.pbz2','')
                subj_electrodes.append(electrode)
                subj_neural_fnames.append(file)

        # Iterate through phases.
        # Day 0 contains Phases A and B, which are encoding and same-day recall, respectively
        # Day 1 contains Phase A (which is different from Day 0's phase A), which is next-day recall.
        phases = [['A','B'] if day == 0 else ['A']][0]
        phases = ['C'] if day == 1 and subj_id == 'e0015TJ' else phases # One subj has a different naming convention
        
        # Iterate through the phases we just defined
        for phase in phases:
            
            # Determine behavior and event filenames
            subj_behav_file = f"{params.path_read}/{dir}/{dir[:-2]}{phase}.mat"
            subj_event_file = f"{params.path_read}/{dir}/{dir[:-2]}Events.pbz2"
            
            # Create a label to better signify the current phase
            if day == 0 and phase == 'A':
                phase_dict = 'Encoding'
            elif day == 0 and phase == 'B':
                phase_dict = 'SameDayRecall'
            elif day == 1:
                phase_dict = 'NextDayRecall'

            # Print report
            print(f"\nID: {subj_id}")
            print(f"Day: {day}")
            print(f"Phase: {phase_dict}")

            # Load and navigate the data using Bryan Zheng's functions
            bhv = read_bhv(subj_behav_file)
            fs, codes, trials = find_trials(subj_event_file)
            matched_trials = match_trials(bhv, codes, trials, fs)
        
            ## Save info:
            matched_trials_all.append(matched_trials)
            df.loc[len(df.index)] = [subj_id , dir.split('_')[-1], dir, dir.split('_')[0], phase_dict]  


    # Create and save event files, save locations in subdir_data column of session_info.csv frame
    path_subj = []
    path_sess = []
    for (row_num, row), matched_trials in zip(df.iterrows(), matched_trials_all):

        ## Create subject-session directory:
        if params.use_subdirs:
            subdir = os.path.join(row["participant_id"], row["session"])
        else:
            subdir = ''

        # Create subject and session paths
        path_subj.append(os.path.join(params.path_save, row["participant_id"]))
        path_sess.append(os.path.join(params.path_save, subdir))
        
        # Create directory
        os.makedirs(path_sess[-1], exist_ok = True)
        
        # Create events list & convert to dataframe
        events = np.concatenate([trials for trials in matched_trials.values()])
        events = pd.DataFrame(events).rename(columns={0: 'code', 1: 'time'})

        # Event-file name
        event_file = os.path.join(path_sess[-1], row["participant_id"] + "_" + row["session"] + "_events.csv")

        # Save event file
        events.to_csv(event_file, index = False)

    # Insert subdir location and save session info
    df["path_sess"] = path_sess
    df['path_subj'] = path_subj
    df.to_csv(os.path.join(params.path_save, "session_info.csv"), index = False)

    # Give the session info back
    return df


def make_raws(session_info, params):
    '''
    Functions:
    - Reads in all SEEG timeseries data and saves a file per subject per session.
    - Keeps channel info in register with SEEG data and saves as corresponding csv.
    '''

    # We don't need these imports around generally
    from joblib import Parallel, delayed
    from src.preproc.utils import construct_raw

    # Parallelization wrapper
    def parallel(row, params):
        """ read raw timeseries for a single session. also reads metadata about channels, in chinfo. """
        
        # 
        raw, chinfo = construct_raw(row, params)
        
        # Filename for this session
        fname_base = row["participant_id"] + "_" + row["session"]

        # Save .fif file
        raw.save(os.path.join(row['path_sess'], fname_base + "_raw.fif"), overwrite = True)
        
        # Save channel info
        chinfo.to_csv(os.path.join(row['path_subj'], row["participant_id"] + "_chinfo.csv"), index = False)
        
        return raw, chinfo

    # Parallelize 
    raws = Parallel(n_jobs = params.n_jobs)(delayed(parallel)(row, params) for i, row in session_info.iterrows())

    return raws


def inspect_sessions(session_info, params):
    # - It also marks bad channels and bad time samples, and saves the bad channel info (_chinfo).
    # - Marking bad channels is done manually, using MNE popups. Close the plot to let the script proceed. 
    # - Bad channels are saved to the _chinfo file and the raw file.

    # - The best way to use mne's interactive graphical windows on oscar is (unfortunately) via virtual desktop
    #   on OOD. Displaying larger files in the popup window can be slow -- potentially plotting onnly a subset
    #   of timepoints or channels can speed this up.

    import time

    for i, row in session_info.iterrows():
        print("Marking bads for subject " + row["participant_id"] + ", session " + row["session"] + "...")
        print("File " + str(i + 1) + " of " + str(len(session_info)))

        ## get paths:
        fname_base = row["participant_id"] + "_" + row["session"]
        fname_fif  = os.path.join(row['path_sess'], fname_base + "_raw.fif")

        # read raw data
        raw = mne.io.read_raw_fif(fname_fif, preload = True)

        # Read channel info
        chinfo = pd.read_csv(os.path.join(row['path_subj'], row["participant_id"] + "_chinfo.csv"))

        if params.do_apply_bads:
            print("Applying saved bads and skipping ploting...")
            # Default is no bads, otherwise apply bads if specified
            raw.info["bads"] = params.bads.get(row["participant_id"] + "_" + row["session"], [])
        else:
            ## Manually mark bad channels and update each subject's chinfo
            raw.compute_psd().plot()
            raw.plot(block = True)  ## halts loop until plot is closed; use to mark bads.
            
        ## Save bad channel info
        ## Unlike _chinfo.csvs, this csv is written to session subdirectories, as bad channels may differ across sessions.
        ## File also appended with date and time to avoid overwriting and losing prior bad channel markings.
        ## Even so, this file is mainly for record-keeping and is not anticipated to be used substantively by downstream analyses
        ## (as the bad channels are already marked in the raw file).
        chinfo["is_bad"] = [ch in raw.info["bads"] for ch in chinfo["contact"]]
        fname_out = os.path.join(row['path_sess'], "badchs_" + time.strftime("%Y%m%d-%H%M%S") + ".csv")
        chinfo[["index", "contact", "is_bad"]].to_csv(fname_out, index = False)
        
        # Overwrite the raw file.
        # NB: There is no data-loss here, so if another definition of bad channels is desired, this can be modified within the raw.
        raw.save(os.path.join(row['path_sess'], fname_base + "_raw.fif"), overwrite = True)


def preproc_sessions(session_info, params):
    '''
    Functions:
    - Attenuates power-line noise, re-references, bandpass filters, detects outliers
    - Settings are all in params.
    - Saves data in .fif and .set formats (MNE, EEGLab)
    - If line-noise persists (see figs), you can increase num dims in the session_info.csv file.
    '''
    from src.preproc.utils import remove_line_noise, rereference, annot_bad_times

    for i, row in session_info.iterrows():
        print("Processing subject " + row["participant_id"] + ", session " + row["session"] + "...")
        print("File " + str(i + 1) + " of " + str(len(session_info)))

        # This gets extended based on what steps are done
        fname = row["participant_id"] + "_" + row["session"] + '_raw'

        # Read raw file
        raw = mne.io.read_raw_fif(os.path.join(row['path_sess'], fname + ".fif"), preload = True)
        
        # Read channel info file
        chinfo = pd.read_csv(os.path.join(row['path_subj'], row["participant_id"] + "_chinfo.csv"))


        # Remove line noise
        if params.do_rmline:
            # Default is 3, but use specified value if it exists:
            nremove = params.n_components_60hz.get(row["participant_id"] + "_" + row["session"], 3)
            raw = remove_line_noise(raw, nremove = nremove)
            fname += '_no60hz'
            if params.save_step_rmline:
                save_raw_if(raw, params, row['path_sess'], fname)
                save_plt_if(raw, params, fname)

        # Re-referencing
        if params.do_rerefing:
            raw, chinfo = rereference(raw, chinfo, method = params.reref_method)
            fname += '_ref'
            if params.save_step_rerefing:
                save_raw_if(raw, params, row['path_sess'], fname)
                save_plt_if(raw, params, fname)

        # Bandpass filter data
        if params.do_bandpass:
            raw = raw.filter(params.bandpass[0], params.bandpass[1])
            fname += '_bp'
            if params.save_step_bandpass:
                save_raw_if(raw, params, row['path_sess'], fname)
                save_plt_if(raw, params, fname)

        ## Annotate bad time samples
        if params.do_rmouts:
            bad = annot_bad_times(raw, thresh = 20, consensus = 1, method = "mad", duration = 6/1024)
            raw.set_annotations(raw.annotations + bad)
            fname += '_rmouts'
            
            # Always want a copy of final, full-sample raw
            save_raw_if(raw, params, row['path_sess'], fname)

        # Downsample the data
        if params.do_downsample_session:
            raw = raw.resample(params.sample_freq_new)
            fname += '_ds'

            # Last step if applied, so no save flag
            save_raw_if(raw, params, row['path_sess'], fname)
            save_plt_if(raw, params, fname)


def clip_sessions(session_info, params):
    ''' Like epoching, but without the equal-size restriction.'''
    from src.preproc.utils import get_times_from_notes, clip_iterator

    # 
    for i, row in session_info.iterrows():

        # Say what subject & file we're on
        print("Processing subject " + row["participant_id"] + ", session " + row["session"] + "...")
        print("File " + str(i + 1) + " of " + str(len(session_info)))
        
        # Read raw file - preferably one that isn't downsampled
        fname = row["participant_id"] + "_" + row["session"]
        raw = mne.io.read_raw_fif(os.path.join(row['path_sess'], fname + "_raw_no60hz_bp_rmouts.fif"), preload = True)
        
        # Get trial starts and ends
        trial_times = get_times_from_notes(raw, 'trial_start', 'trial_stop')

        # Clip out data prior to first event
        if params.do_clip_pre:
            new = raw.copy()
            new.crop(0, trial_times[0,0])
            save_raw_if(new, params, row['path_sess'], fname + "_pre")
        
        # Clip out "body" of data
        if params.do_clip_dur:
            new = raw.copy()
            new.crop(trial_times[0,0]-1, trial_times[-1,-1]+1)
            save_raw_if(new, params, row['path_sess'], fname + "_dur")

        # Clip out data after last event
        if params.do_clip_post:
            new = raw.copy()
            new.crop(trial_times[-1,-1], raw.times[-1])
            save_raw_if(new, params, row['path_sess'], fname + "_post")


        # Clip out trials
        clip_iterator(raw, params, str_beg = 'trial_start', str_end = 'trial_stop', path = row['path_sess'], fname = fname, sfx = 'trial')

        # Clip out first movie presentations
        clip_iterator(raw, params, str_beg = 'clip_start', str_end = 'clip_stop', path = row['path_sess'], fname = fname, sfx = 'clip')

        # Broken...
        # clip_iterator(raw, params, str_beg = 'clipcue_start', str_end = 'clipcue_stop', path = row['path_sess'], fname = fname, sfx = 'clip')

        # Clip out first movie presentations
        clip_iterator(raw, params, str_beg = 'loc_start', str_end = 'loc_resp', path = row['path_sess'], fname = fname, sfx = 'loc')



def epoch_sessions(session_info, params):
    '''
    This script reads preprocessed raw timeseries data and epochs it into trials.
    - Epoch metadata (incl triggers, events), saved in a csv file with one row per trial
    - NB: For subsequent analyses using the epoched data, you can bind the behavioral data (behavioral_data.csv),
          with the seeg data using the trial_num column in the metadata file.
    - NB: There may be fewer trials in epochs than in behavioral data, if some trials were annotated as bad.
    '''

    # Load behavioral data
    beh_data = pd.read_csv(os.path.join(params.path_save, "behavioral_data.csv"))

    log = []
    for i, row in session_info.iterrows():

        # Say what subject & file we're on
        print("Processing subject " + row["participant_id"] + ", session " + row["session"] + "...")
        print("File " + str(i + 1) + " of " + str(len(session_info)))

        # Create an epoch files directory
        dir_subj_epochs = os.path.join(row['path_save'], "epochs", row["participant_id"], row["session"])
        os.makedirs(dir_subj_epochs, exist_ok = True)
        
        # Read raw file 
        fname_base = row["participant_id"] + "_" + row["session"]
        raw = mne.io.read_raw_fif(os.path.join(row['path_sess'], fname_base + "_raw_final.fif"), preload = True)
        
        # Get events from annotations
        events, event_id = mne.events_from_annotations(raw, event_id = params.trigger_from_desc_dict)
        
        # For each epoch, select data and save as a file
        for epoch_type in params.epoch_info.keys():
            
            ## create metadata df from events/trigs:
            metadata, events_out, event_id_out = mne.epochs.make_metadata(
                events = events,
                event_id = event_id,
                tmin = params.epoch_info[epoch_type]["metadata_tmin"],
                tmax = params.epoch_info[epoch_type]["metadata_tmax"],
                sfreq = raw.info["sfreq"],
                row_events = params.epoch_info[epoch_type]["row_events"]
            )

            ## subset beh_data to this subject and session:
            is_sess = (beh_data["participant_id"] == row["participant_id"]) & (beh_data["session"] == row["session"])
            beh_data_sub = beh_data.loc[is_sess, :]

            ## bind columns of beh_data_sub and metadata, only if there are the same number of rows.
            ## Otherwise, skip this subject*session, and raise error at end.
            if beh_data_sub.shape[0] != metadata.shape[0]:
                err_msg = "n trials: beh = " + str(beh_data_sub.shape[0]) + ", trigs = " + str(metadata.shape[0])
                log.append(err_msg)
                continue
            else:
                log.append("ok")
            
            metadata["trial_num"] = beh_data_sub["trial_num"].values  ## NB: ASSUMES BOTH BEH AND EPOCHS ARE CHRONOLOG.
            
            ## epoch and save
            epochs = mne.Epochs(
                raw,
                baseline = None,
                detrend = None,
                events = events_out,
                event_id = event_id_out,
                tmin = params.epoch_info[epoch_type]["tmin"],
                tmax = params.epoch_info[epoch_type]["tmax"],
                preload = True,
                metadata = metadata,
                reject_by_annotation = False  ## this rejects epochs that were annotated as "BAD_outlier" w/in clean_raws.py
                )
            
            # Save epoch files
            fname_epochs = fname_base + + "_no60hz_ref_bp_" + epoch_type + "-epo"
            save_raw_if(epochs, params, dir_subj_epochs, fname_epochs)

            ## save metadata
            metadata.to_csv(fname_epochs + "-metadata.csv")


    session_info['log'] = log
    print(session_info)
    if any([x != "ok" for x in log]):
        raise ValueError("Some epochs were misaligned!")