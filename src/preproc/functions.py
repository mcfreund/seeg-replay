'''
This file contains the main preprocessing functions.
'''

import os
import mne
import numpy  as np
import pandas as pd
import ipdb
import dill
from joblib import Parallel, delayed
from src.preproc.utils import *
from src.shared.utils  import *

# This function does several things which should be disaggregated:
# - read behavioral data and write an events.csv file
# - write a session_info.csv file tracking the file structure

def make_session_info(params, paths):
    '''
    This function writes a session_info.csv file containing session metadata, which is used in preprocessing.

    It is adapted to some extent from:
    https://github.com/Brainstorm-Program/brainstorm_challenge_2024/blob/main/scripts/brainstorm_reorganize_data.py
    '''
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
        #'2023-02-07_e0016YR_02'
        ]

    # Corrosponding days
    subj_days = [0,1, 0,1, 0,1, 0,1, 0,1]#, 0]

    # For each participant session, collect session info into frame
    for day, dir in zip(subj_days, subj_dirs):

        # Display which subject is being processed
        print(f"\nParticipant Day, Directory: {day}, {dir}")

        # Get subject ID and all files in directory
        subj_id = dir.split('_')[-2]
        subj_fnames = os.listdir(f"{paths.unproc_data}/{dir}")
        
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
            subj_behav_file = f"{paths.unproc_data}/{dir}/{dir[:-2]}{phase}.mat"
            subj_event_file = f"{paths.unproc_data}/{dir}/{dir[:-2]}Events.pbz2"
            
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

            # Copy of behavior .mat file, dictionary format, trial num keys
            bhv = read_bhv_matfile(subj_behav_file)

            # Summarize the timing information from these
            time_info = summarize_behavior_timing(bhv)

            # Get all the trial codes and sample times from the event files
            trials = get_trials_from_event_file(subj_event_file)

            # Fuck everything up. Oh wait, no, don't.
            # matched_trials = match_trials(bhv, codes, trials, fs, verbose=True)

            # Get the index-shift into neural trials to use
            use_shift = check_behavior_against_neural_timing(time_info, trials)
            ind_end   = use_shift + len(bhv)

            # Print basic information so we know what we're getting...
            print(f'Trials in event file: {len(trials)}')
            print(f'Trials in behavior  : {len(bhv)}  ')

            # Subset trials and inform
            if day == 0 and phase == 'A':
                matched_trials = trials[use_shift:ind_end]
                print(f'Interpreting trials {use_shift}-{ind_end} as Encoding')

            elif day == 0 and phase == 'B':
                matched_trials = trials[use_shift:ind_end]
                print(f'Interpreting {use_shift}-{ind_end} as SameDayRecall')

            elif day == 1 and phase == 'A':
                matched_trials = trials[use_shift:ind_end]
                print(f'Interpreting {use_shift}-{ind_end} as NextDayRecall')

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
        path_subj.append(os.path.join(paths.save_preproc, row["participant_id"]))
        path_sess.append(os.path.join(paths.save_preproc, subdir))
        
        # Create directory
        os.makedirs(path_sess[-1], exist_ok = True)
        
        # Create events list, filename, and save. Each event is (code, sample number).
        events     = np.concatenate(matched_trials)
        event_file = os.path.join(path_sess[-1], row["participant_id"] + "_" + row["session"] + "_events.pt")
        dill_save(events, event_file)

    # Insert subdir location and save session info
    df["path_sess"] = path_sess
    df['path_subj'] = path_subj
    df.to_csv(os.path.join(paths.save_preproc, "session_info.csv"), index = False)

    # Give the session info back
    return df


def make_raws(session_info, params, paths):
    '''
    Functions:
    - Reads in all SEEG timeseries data and saves a file per subject per session.
    - Keeps channel info in register with SEEG data and saves as corresponding csv.
    '''

    # Parallelization wrapper
    def parallel(row, params, paths):
        """ read raw timeseries for a single session. also reads metadata about channels, in chinfo. """
        
        # 
        raw, chinfo = construct_raw(row, params, paths)
        
        # Filename for this session
        fname_base = row["participant_id"] + "_" + row["session"]

        # Save .fif file
        raw.save(os.path.join(row['path_sess'], fname_base + "_raw.fif"), overwrite = True, verbose='Error')
        
        # Save channel info
        chinfo.to_csv(os.path.join(row['path_subj'], row["participant_id"] + "_chinfo.csv"), index = False)
        
        return raw, chinfo

    # Parallelize 
    raws = Parallel(n_jobs = params.n_jobs)(delayed(parallel)(row, params, paths) for i, row in session_info.iterrows())
    
    # Local alternate
    #for i, row in session_info.iterrows():
    #    raws = parallel(row, params, paths)

    return raws


def construct_raw(session_row, params, paths):
    """ Constructs raw data object for a given day. Also loads metadata about channels, in chinfo. 
    NB: only the intersection of channel names in chinfo["channel"] and names of the SEEG day subdirectory is kept.
    I.e., we keep only channels for which we have both data and their location (assuming no typos in names).
    """
    subj = session_row['participant_id']
    sess = session_row['session']
    dir  = session_row['subdir_orig']

    # Read channel information
    chinfo = read_subj_channel_info(subj, paths)

    # Read SEEG data in intersection of contacts data files
    data, contacts = load_session_seeg(dir = paths.unproc_data + '/' + dir, contacts = chinfo["contact"])

    # Keep only contacts from data in chinfo
    chinfo = chinfo[chinfo["contact"].isin(contacts)]
    
    # Read events file (from processed data path) and format for MNE
    events = dill_read(os.path.join(session_row['path_sess'], subj + "_" + sess + "_events.pt"))
    events = np.stack([events[:,1], np.zeros_like(events[:,0]), events[:,0]]).T
    
    # Debugging checkpoint 1
    #dill_save(data,'./data/chkpts/data_chkpt_cnstrct_1.pt')

    # Rescale signal
    signals = np.stack([d["signal"] for d in data]) / params.scale # ideally, in V

    # Create MNE metadata
    n_channels = len(data)
    #ch_names  = chinfo["contact"].tolist()
    ch_names   = [d['label'] for d in data ] # this preserves order properly!!!
    ch_types   = ["seeg"] * n_channels

    # Create MNE structure and add line frequency to it
    info = mne.create_info(ch_names, ch_types = ch_types, sfreq = params.sample_freq_native)
    info["line_freq"] = params.line_freq

    # Debugging checkpoint 2
    # dill_save(signals, './data/chkpts/data_chkpt_cnstrct_2.pt')

    ## Construct MNE "raw" type (ensure signals and stim data order match ch_types/names)
    raw = mne.io.RawArray(signals, info)
    
    ## Add events as annotations:
    annots = mne.annotations_from_events(events, params.sample_freq_native, event_desc = params.trigger_to_desc_dict)
    raw.set_annotations(annots)
    
    return raw, chinfo


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


def preproc_sessions(session_info, params, paths):
    '''
    Functions:
    - Attenuates power-line noise, re-references, bandpass filters, detects outliers
    - Settings are all in params.
    - Saves data in .fif and .set formats (MNE, EEGLab)
    - If line-noise persists (see figs), you can increase num dims in the session_info.csv file.
    '''

    for i, row in session_info.iterrows():
        print("Processing subject " + row["participant_id"] + ", session " + row["session"] + "...")
        print("File " + str(i + 1) + " of " + str(len(session_info)))

        # This gets extended based on what steps are done
        fname = row["participant_id"] + "_" + row["session"]

        # Read raw file
        raw = mne.io.read_raw_fif(os.path.join(row['path_sess'], fname + "_raw.fif"), preload = True, verbose='Error')
        
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
                save_plt_if(raw, params, fname, paths)

        # # Bipolar, for reference
        # if params.do_rerefing:
        #     copy = raw.copy()
        #     copy, chinfo_copy = rereference(copy, chinfo, row['path_sess'], method = 'bipolar')
        #     if params.save_step_rerefing:
        #         save_raw_if(copy, params, row['path_sess'], fname + '_bip')
        #         save_plt_if(copy, params, fname, paths)

        # # Unipolar, for reference
        # if params.do_rerefing:
        #     copy = raw.copy()
        #     copy, chinfo_copy = rereference(copy, chinfo, row['path_sess'], method = 'unipolar')
        #     if params.save_step_rerefing:
        #         save_raw_if(copy, params, row['path_sess'], fname + '_uni')
        #         save_plt_if(copy, params, fname, paths)

        #         # Conserve memroy
        #         del copy
        
        # # Laplacian w/ self-ref, for reference
        # if params.do_rerefing:
        #     copy = raw.copy()
        #     copy, chinfo_copy = rereference(copy, chinfo, row['path_sess'], method = 'laplacian', selfref_first=True)
        #     if params.save_step_rerefing:
        #         save_raw_if(copy, params, row['path_sess'], fname + '_srf')
        #         save_plt_if(copy, params, fname, paths)

        # Re-referencing
        if params.do_rerefing:
            raw, chinfo = rereference(raw, chinfo, row['path_sess'], method = params.reref_method)
            fname += '_ref'
            if params.save_step_rerefing:
                save_raw_if(raw, params, row['path_sess'], fname)
                save_plt_if(raw, params, fname, paths)

        # Bandpass filter data
        if params.do_bandpass:
            raw = raw.filter(params.bandpass[0], params.bandpass[1])
            fname += '_bp'
            if params.save_step_bandpass:
                save_raw_if(raw, params, row['path_sess'], fname)
                save_plt_if(raw, params, fname, paths)

        ## Annotate bad time samples
        if params.do_rmouts:
            bad = annot_bad_times(raw, thresh = 20, consensus = 1, method = "mad", duration = 6/1024)
            raw.set_annotations(raw.annotations + bad)
            fname += '_rmouts'
            
        # Always want a copy of final, full-sample raw
        save_raw_if(raw, params, row['path_sess'], fname)

        # Downsample the data
        if params.do_downsample_session:
            # Extract event triggers to jointly resample them to same time-grid as raw data.
            # see: https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.resample
            # Use regexp = None so that '^BAD' annots are preserved.
            events, event_desc = mne.events_from_annotations(raw, event_id = params.trigger_from_desc_dict, regexp = None)
            raw, events = raw.resample(params.sample_freq_new, events = events)
            ## Add events back to raw as annotations
            annots = mne.annotations_from_events(events, raw.info['sfreq'], params.trigger_to_desc_dict)
            raw.set_annotations(annots)
            fname += '_ds'
            # Last step if applied, so no save flag
            save_raw_if(raw, params, row['path_sess'], fname)
            save_plt_if(raw, params, fname, paths)


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
        raw = mne.io.read_raw_fif(
            os.path.join(row['path_sess'], fname + params.suffix_preproc + "_raw.fif"), preload = True)
        
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



def time_frequency_decompose(session_info, params, paths):
    '''
    This function reads preprocessed raw timeseries data and implements a morelet wavelet transform, 
    then saves the time-frequency power timeseries.
    '''

    from src.preproc.utils import time_frequency_decomp, tfarray_to_raw, bin_wavelets
    
    for i, row in session_info.iterrows():

        path_sess = os.path.join(paths.processed_raws, row["participant_id"], row["session"])

        # Say what subject & file we're on
        print("Processing subject " + row["participant_id"] + ", session " + row["session"])
        print("File " + str(i + 1) + " of " + str(len(session_info)))

        # Read raw file (and drop bads)
        
        fname_base = os.path.join(path_sess, row["participant_id"] + "_" + row["session"] + params.suffix_preproc)
        raw = mne.io.read_raw_fif(fname_base + "_raw.fif", preload = True)
        raw.drop_channels(raw.info["bads"])
        chinfo = pd.read_csv(os.path.join(row['path_subj'], row["participant_id"] + "_chinfo.csv"))
        
        ## prefix for output filename:
        #fname_out = row["participant_id"] + "_" + row["session"]
        
        ## subset channels
        is_greymatter = chinfo["White Matter"] == 0
        
        for region_i, region in enumerate(params.tf_regions):

            is_region = chinfo["Level 3: gyrus/sulcus/cortex/nucleus"].str.contains(region)
            contacts = chinfo[is_region & is_greymatter]["contact"].tolist()
            contacts = [c for c in contacts if c in raw.ch_names] ## only keep contacts that are in the raw file
            raw_region = raw.copy().pick(contacts)

            ## apply time-frequency decomp

            tf_array = time_frequency_decomp(raw_region, params)
            tf_binned = bin_wavelets(tf_array, params)  ## aggregate into bands
            
            ## put into raw objects
            
            raw_tf = tfarray_to_raw(tf_array,
                params.tf_freqs, raw_region.ch_names, "seeg", 
                raw_region.info["sfreq"], raw_region.annotations)

            raw_binned = tfarray_to_raw(tf_binned,
                params.tf_bands.keys(), raw_region.ch_names, "seeg", 
                raw_region.info["sfreq"], raw_region.annotations)

            ## save

            fname_out_tf = fname_base + "_morletfull_" + params.tf_regions_fnames[region_i]
            fname_out_binned = fname_base + "_morletbins_" + params.tf_regions_fnames[region_i]
            if params.save_fif:
                save_raw_if(raw_tf, params, path_sess, fname_out_tf)
                save_raw_if(raw_binned, params, path_sess, fname_out_binned)



def epoch_sessions(session_info, params, paths):
    '''
    This script reads preprocessed raw timeseries data and epochs it into trials.
    - Epoch metadata (incl triggers, events), saved in a csv file with one row per trial
    - NB: For subsequent analyses using the epoched data, you can bind the behavioral data (behavioral_data.csv),
          with the seeg data using the trial_num column in the metadata file.
    - NB: There may be fewer trials in epochs than in behavioral data, if some trials were annotated as bad.
    '''
    
    # Check for bad configurations of preproc parameters
    if "_ds" in params.suffix_preproc and params.do_downsample_epochs:
        raise ValueError("Don't generate downsampled epochs from already-downsampled raws.")
    
    # Load behavioral data
    beh_data = pd.read_csv(os.path.join(paths.save_preproc, "behavioral_data.csv"))

    log = []
    for i, row in session_info.iterrows():

        # Say what subject & file we're on
        print("Processing subject " + row["participant_id"] + ", session " + row["session"] + "...")
        print("File " + str(i + 1) + " of " + str(len(session_info)))

        # Create an epoch files directory
        #dir_subj_epochs = os.path.join(ppaths.save_preproc, "epochs", row["participant_id"], row["session"])
        #os.makedirs(dir_subj_epochs, exist_ok = True)
        
        # Read raw file 
        fname_base = os.path.join(
            row['path_sess'], row["participant_id"] + "_" + row["session"] + params.suffix_preproc)
        raw = mne.io.read_raw_fif(fname_base + "_raw.fif", preload = True)
        
        # Get events from annotations
        events, event_id = mne.events_from_annotations(raw, event_id = params.trigger_from_desc_dict)
        
        # For each epoch, select data and save as a file
        for epoch_type in params.epoch_info.keys():

            is_bad_combo = (epoch_type == "clipcue") & (row["session"] != "Encoding")
            if is_bad_combo:
                ## clipcue only present during encoding session, therefore skip.
                continue
            
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
            
            beh_data_sub = beh_data_sub.sort_values(by = "trial_num")
            metadata["trial_num"] = beh_data_sub["trial_num"].values  ## NB: ASSUMES BOTH BEH AND EPOCHS ARE CHRONOLOG.
            
            ## epoch and save
            epochs_list = []
            # Anti-aliasing filter and decimation factor
            if params.do_downsample_epochs:
                # https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html#best-practices
                raw.filter(0, params.sample_freq_new / 3, n_jobs = params.tf_n_jobs)
                decim = int(params.sample_freq_native / params.sample_freq_new)
            else:
                decim = 1  ## no downsampling
                
            for reject_bad_epos in [True, False]:
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
                    decim = decim,
                    ## if true, this rejects epochs that were annotated as "BAD_outlier" w/in raw file:
                    reject_by_annotation = reject_bad_epos
                    )
                epochs_list.append(epochs)
            epochs = epochs_list[1]  ## preserve all epochs
            is_bad = [x != () for x in epochs_list[0].drop_log] ## but save markers of those dropped, and add this to metadata.
            metadata["has_extreme_val"] = is_bad  ## this keeps flexibility for downstream analyses
            ## NB: currently in mne, epoching twice like this seems to be simplest way to preserve all epochs while still ID'ing
            ## those marked bad.
            
            # Save epoch files
            fname_epochs = os.path.join(paths.save_preproc, fname_base + "_" + epoch_type + "-epo")
            save_epochs(epochs, metadata, fname_epochs, params)

    if any([x != "ok" for x in log]):
        print(log)
        raise ValueError("Some epochs were misaligned!")
