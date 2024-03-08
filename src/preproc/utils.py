import os
import re
import mne
import pickle
import meegkit
import scipy
import scipy as sp
import ipdb


import pandas as pd
import scipy.io as sio
import numpy as np
import itertools
import bz2
import _pickle

from datetime import datetime

from src.shared.utils import *

def load_pkl(path):
    """ loads compressed pickle file """

    with bz2.open(path, 'rb') as f:
        neural_data = _pickle.load(f)
    return neural_data


def get_trials_from_event_file(event_file):
    """ finds *all* trials with associated codes in an events file """
    events = load_pkl(event_file)

    # Get (event code, sample number) event tuples
    signals = events['signal']

    # The relevant codes are
    # 20: "fix_start",
    # 22: "fix_resp",
    # 72: "clip_start",
    # 74: "clip_stop",
    # 82: "clipcue_start",  # This is only in encoding trials
    # 84: "clipcue_stop",   # This is strangely in recall trials
    # 36: "loc_start",
    # 38: "loc_resp",
    # 56: "col_start",
    # 58: "col_resp",

    # List of relevant codes:
    relevant_codes = np.array([9, 20, 22, 72, 74, 82, 84, 36, 38, 56, 58, 18])

    # Organize tuples into trials based on codes
    trials  = []
    ignored = []
    trial_num = 0
    prev_code = None
    for pair in signals:
        # Trial code is first element, time is second
        code = pair[0]
        
        # Start a new trial if we see a 9
        if code == 9:

            # Start new trial, code list, and ignored list
            trials.append([pair])
            ignored.append([])
        
        elif code in relevant_codes:
            # Trials may start without a start code, so check if we skipped one or more
            if prev_code != None:
                skipped = np.where(code == relevant_codes) <= np.where(prev_code == relevant_codes)
            else:
                skipped = True

            # If we skipped a start code, need to init a new trial list
            if skipped:
                trials.append([pair])
                ignored.append([])
                skipped = False

            # If we didn't start a new trial, organize next code under current trial
            trials[-1].append(pair)

        elif len(trials) > 0:
            # There are codes e.g. 16 and 27 that we have no info on...
            ignored[-1].append(code)

        # Save whatever the previous code was
        prev_code = code
        
    #print(f'List of codes ignored by trial:')
    #print(ignored)

    #### Quality control trial blocks, turn them into arrays ###
    
    # Check all trials are strictly in order
    times = [trial[0][1] for trial in trials]
    in_order = [time > prev for time, prev in zip(times[1:],times[0:-1])]
    if ~np.all(in_order): print(f'Trial start times are not in order!')

    for i, trial in enumerate(trials):

        # Check all the trials have all the codes we care about
        missing = []
        for code in relevant_codes:
            if code not in [pair[0] for pair in trial]:
                missing.append(code)

        # Check all codes are strictly in order
        times = [pair[1] for pair in trial]
        in_order = [time > prev for time, prev in zip(times[1:],times[0:-1])]
        if ~np.all(in_order): print(f'Trial {i+1} code times are not in order!')

        # Notify user of missing and ignored codes
        if len(missing) > 0 or len(ignored[i]) > 0:
            print(f'Trial {i+1} has missing and ignored {missing}, {ignored[i]}')

        # Convert to array, 
        trials[i] = np.array(trial)
        trials[i][:,1] = trials[i][:,1]

    # Return sampling frequency and trial list
    return trials


def read_subj_channel_info(subj, paths):
    """ wrangles channel / anatomical info from excel files. optionally sorts row order to match channel order in a signal_labels object."""

    # Read channel information file
    chinfo = pd.read_excel(os.path.join(paths.chnl, subj, "parsed.xlsx"))

    ## Create new columns in chinfo that separate electrode and site/contact info:
    ## NB: sites nested in electrodes
    chinfo[["electrode", "site"]] = chinfo["contact"].str.extract('([a-zA-Z-]+)(\d+)', expand = True)
    chinfo = chinfo.sort_values("index")
    
    ## create col that will become ch_name in raw:
    #chinfo["ch_name"] = chinfo["Anatomical Label"] + "_wm-" + chinfo["White Matter"].astype("string")
    ## exception for formatting:
    if subj == "e0015TJ":
        chinfo["contact"] = chinfo["contact"].str.replace('-', '')
    
    return chinfo

def load_session_seeg(dir, contacts = None):
    """ loads all SEEG data from single session"""

    if contacts is None:
        ## list of all in directory:
        data_files = [os.path.join(dir, file) for file in os.listdir(dir) if '.pbz2' in file and "Events" not in file]
    else:
        ## list of only those specified in contacts. if specified, ensures set and order of data matches chinfo.
        ## find all files that end in c + ".pbz2" for c in contacts:
        ## get all files in filelist that match pattern in contacts:
        data_files = [os.path.join(dir, file) for file in os.listdir(dir) if any([c + '.pbz2' in file for c in contacts])]
        contacts = [contact for contact, file in zip(contacts, data_files) if os.path.exists(file)]
        
    data = [load_pkl(fn) for fn in data_files]

    # Debugging checkpoint
    # dill_save(data, './data/chkpts/data_chkpt_load.pt')

    return data, contacts


# Save selector for raws
def save_raw_if(raw, params, path_sess, fname):
    ''' Function for saving file types selected in params.'''

    # Filename without a filetype suffix
    fname = os.path.join(path_sess, fname)

    # Save .fif format
    if params.save_fif:
        raw.save(fname + '_raw.fif', overwrite = True)

    # Save .set format
    if params.save_set:
        mne.export.export_raw(fname + '.set', raw, overwrite = True)

    # Save .csv format
    if params.save_csv:
        df = raw.to_data_frame()
        df.to_csv(fname + '.csv', index=False)

    # Save .h5 format
    if params.save_h5:
        df = raw.to_data_frame()
        df.to_hdf(fname + 'h5')


# Plot functions
def save_plt_if(raw, params, fname, paths):
    ''' Function for saving plots, if selected in params.'''

    if params.save_plt:
        fig = raw.compute_psd().plot(show = False)
        fig.axes[0].set_title("raw timeseries: " + fname + "\n" + t)
        fn_psd = os.path.join(paths.figs, fname ,'.png')
        fig.savefig(fn_psd)


def summarize_behavior_timing(bhv):
    # Want timing data on every trial to match against neural data
    time_info = pd.DataFrame(columns = ['Trial','TrialStart','Code9Time','Code22Time','Code38Time','Code58Time','Code18Time'])

    # Codes to look for timing informatoin for (should be more diagnostic than start and end times)
    codes     = [9,22,38,58,18]
    code_flds = ['Code9Time','Code22Time','Code38Time','Code58Time','Code18Time']

    # Entries bhv are dictionaries of trial info from read_bhv_matfile and mat_to_dict
    for mat in bhv:
        # Save trials in order, keeping trial nums & relative start times (yes, "absolute" is mislabeled)
        time_info.loc[mat['Trial'], ['Trial']]      = mat['Trial']
        time_info.loc[mat['Trial'], ['TrialStart']] = mat['AbsoluteTrialStartTime']

        # Mask out times that codes occurred, and save a list of code times
        msk = [i in codes for i in mat['BehavioralCodes']['CodeNumbers']]
        time_info.loc[mat['Trial'], code_flds] = mat['BehavioralCodes']['CodeTimes'][msk]

    return time_info

# 
def check_behavior_against_neural_timing(time_info, trials):
    
    # Compute intervals in behavioral timing between trial starts
    intervals = np.diff(time_info['TrialStart'])/1000

    # Get neural data trial start times and intervals between them
    trial_start_samples = [pair[0,1] for pair in trials]
    trial_start_times   = np.array(trial_start_samples)/1024
    trial_start_diffs   = np.diff(trial_start_times)

    # We'll look for the behavioral start-interval profile in the neural data
    n_trials = len(trial_start_diffs)+1
    n_found  = len(intervals) + 1
    n_shifts = n_trials - n_found + 1

    # Check timing discrepancy for each possible index shift:
    error = np.full(n_shifts, np.nan)
    for shift in range(n_shifts):
        error[shift] = np.sum(np.abs(trial_start_diffs[shift:(n_found-1+shift)] - intervals))

    # Report how good each index shift looked
    print(f'Mismatch error by shift index: ')
    print(np.round(error).astype(int))

    # This error really should not be above maybe a second
    assert min(error) < 1
    
    # Determine which shift to use
    use_shift = np.where(error == min(error))[0][0]
    
    # Report
    print(f'Starting match at index {use_shift} with total discrepancy {error[use_shift]:2f}[s].')

    return use_shift



def read_bhv_matfile(subj_behav_file):
    """ load behavioral data from .mat file and extra trial data
        returns dictionary of all behavioral data per trial
    """
    bhv = []

    f = sp.io.loadmat(subj_behav_file, struct_as_record=False, squeeze_me=True)
    for k in f.keys():
        if k[0:5] == 'Trial' and k[5:].isdigit():
            trial = f[k]
            bhv.append(mat_to_dict(trial))

    return bhv


def mat_to_dict(mat):
    """ convert mat_struct object to a dictionary recursively
    """
    dict = {}

    for field in mat._fieldnames:
        val = mat.__dict__[field]
        if isinstance(val, sp.io.matlab.mat_struct):
            dict[field] = mat_to_dict(val)
        else:
            dict[field] = val

    return dict


def remove_line_noise(raw, nremove = 1, nfft = 1024, nkeep = None, blocksize = None):
    ## adapted from: https://mne.discourse.group/t/clean-line-noise-zapline-method-function-for-mne-using-meegkit-toolbox/7407/2
    ## uses: https://github.com/nbara/python-meegkit/blob/9204cfd8d596be479ddae932108445b4f560010a/meegkit/dss.py#L149

    # Copy structure to keep metadata
    raw_out = raw.copy()

    # Remove line noise
    out, artif = meegkit.dss.dss_line(
        X = raw.get_data().T, 
        fline = raw.info["line_freq"],
        sfreq = raw.info['sfreq'],
        nremove = nremove,
        nfft = nfft,
        nkeep = nkeep,
        blocksize = blocksize,
        show = False)

    # Copy output back to raw
    raw_out._data = out.T

    return raw_out #out.T


def _cluster_contacts(signal_list):
    """ return electrode-contact hierarchy """
    ## from: https://github.com/Brainstorm-Program/Brainstorm-Challenge-PreProcessing/blob/main/rereference_bipolar

    signal_dict = {}

    for sig in signal_list:
        sig_grps = re.search('([A-Z|0-9]{1,}[-| ])?([A-Z]{1,})([0-9]{1,})', sig, re.IGNORECASE)
        if sig_grps:
            n_grps = len(sig_grps.groups())
            electrode = ''.join(filter(None,[sig_grps.group(i) for i in range(1, n_grps)]))
            num = sig_grps.group(n_grps)
            if electrode in signal_dict.keys():
                assert int(num) not in signal_dict[electrode]
                signal_dict[electrode].append(int(num))
            else:
                signal_dict[electrode] = [int(num)]

    return signal_dict


def _reref_electrodes(signals, signal_list, method = 'laplacian', selfref_first = False):
    """ references given signals """
    ## adapted from: https://github.com/Brainstorm-Program/Brainstorm-Challenge-PreProcessing/blob/main/rereference_bipolar

    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6495648/
    # signals: 2d array of time-series data (rows = electrodes as defined and indexed by signal list, columns = time)
    # signal_list = electrode labels, in order of rows in signals variable; must be the same length
    # method = referencing method
           
    signal_dict = _cluster_contacts(signal_list) 

    sig_list_new = []
    signals_new = []
    if method == 'unipolar':
        signals_new = signals
    else:
        for sig in signal_dict:
            if method == 'mean':
                channels = [signal_list.index(e) for e in signal_list if sig in e]
                sig_ref = signals[channels, :] - np.mean(signals[channels,:], axis=0)
                signals_new.append(sig_ref)
            
            else:
                minn = np.min(signal_dict[sig])
                maxn = np.max(signal_dict[sig])
                for num in signal_dict[sig]:
                    if method == 'bipolar':
                        if num >= minn and num < maxn:
                            adj = 1
                            if num + adj <= maxn:
                                while not '{}{}'.format(sig, num+adj) in signal_list and num + adj <= maxn:
                                    adj += 1
                                sig_list_new.append('{}{}'.format(sig, num))
                                sig_ref_i = signal_list.index('{}{}'.format(sig, num))
                                sig_ref_j = signal_list.index('{}{}'.format(sig, num + adj))
                                sig_ref = signals[sig_ref_i, :] - signals[sig_ref_j, :]
                                signals_new.append(sig_ref)

                    elif method == 'laplacian':
                        if num > minn and num < maxn:
                            hi = 1
                            lo = 1
                            above = '{}{}'.format(sig, num + hi)
                            below = '{}{}'.format(sig, num - lo)
                            while above not in signal_list or below not in signal_list and num + hi <= maxn:
                                if above not in signal_list:
                                    hi += 1
                                if below not in signal_list:
                                    lo += 1
                                above = '{}{}'.format(sig, num + hi)
                                below = '{}{}'.format(sig, num - lo)

                            sig_ref_i = signal_list.index(below)
                            sig_ref_j = signal_list.index(above)
                            sig_ref_0 = signal_list.index('{}{}'.format(sig, num))

                            print('{} = {} - {}'.format('{}{}'.format(sig, num), below, above))

                            sig_list_new.append('{}{}'.format(sig, num))
                            sig_ref = signals[sig_ref_0, :] - (signals[sig_ref_i, :] + signals[sig_ref_j, :])/2
                            signals_new.append(sig_ref)
                        elif num == minn and selfref_first:
                            hi = 1
                            above = '{}{}'.format(sig, num + hi)
                            while above not in signal_list and num + hi <= maxn:
                                hi += 1
                                above = '{}{}'.format(sig, num + hi)

                            sig_ref_j = signal_list.index(above)
                            sig_ref_0 = signal_list.index('{}{}'.format(sig, num))

                            print('{} = {} - {}'.format('{}{}'.format(sig, num), '{}{}'.format(sig, num), above))

                            sig_list_new.append('{}{}'.format(sig, num))
                            sig_ref = signals[sig_ref_0, :] - signals[sig_ref_j, :]
                            signals_new.append(sig_ref)                                

    if len(sig_list_new) == 0:
        sig_list_new = signal_list

    signals_new = np.vstack(signals_new)

    return signals_new, sig_list_new


def rereference(raw, chinfo, path_sess, method = 'laplacian', drop_bads = True, selfref_first = False):
    """ wrapper for feeding mne objects into  _ref_electrodes() """

    raw_copy = raw.copy()
    if drop_bads:
        raw_copy.drop_channels(raw_copy.info["bads"])
    
    data_ref, ch_names_ref =_reref_electrodes(
        signals = raw_copy.get_data(),
        signal_list = raw_copy.ch_names,
        method = method,
        selfref_first = selfref_first)

    if method == "laplacian":
        ## mark chinfo with those that were rereferenced
        chinfo["survived"] = chinfo["contact"].isin(ch_names_ref)
        ## Save csv file of those that survived. As with the csv written by inspect_raws(), this is primarily for 
        ## record keeping and is not expected to be used by downstream analyses.
        fname_out = os.path.join(path_sess, "survived_laplacian_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv")
        chinfo[["index", "contact", "survived"]].to_csv(fname_out, index = False)

    ## add data back to raw
    dropped_chs = [ch for ch in raw_copy.ch_names if ch not in ch_names_ref]  ## drop
    raw_copy.drop_channels(dropped_chs)
    if not all(np.sort(raw_copy.ch_names) == np.sort(ch_names_ref)):
        raise Exception("ch_names_ref and raw_copy.ch_names do not match.")
    raw_copy[ch_names_ref] = data_ref
    
    return raw_copy, chinfo



def _get_bad_samples(data, thresh = 20, consensus = 1, method = "mad"):
    ''' adapted from https://github.com/Brainstorm-Program/Brainstorm-Challenge-PreProcessing/blob/main/artifact_rejection '''
    if method == "mad":
        mads = scipy.stats.median_abs_deviation(data, axis = 1, scale = "normal")  ## is of length n_channels
        resids = (np.abs(data.T - np.median(data, axis = 1)) / mads).T
        is_bad = resids > thresh
    elif method == "zscore":
        is_bad = np.abs(scipy.stats.zscore(data, axis = 1)) > thresh
    reject = np.sum(is_bad, axis = 0) > (consensus - 1)
    print("Marked " + str(round(np.mean(reject), 4)) + "% samples bad.")

    return reject


def annot_bad_times(raw, description = "BAD_outlier", thresh = 20, duration = 1, consensus = 1, method = "mad"):

    bad_samp = _get_bad_samples(raw.get_data(), thresh = thresh, consensus = consensus, method = method)
    bad_times = raw.times[bad_samp] - duration/2  ## center
    bad_annots = mne.Annotations(
        onset = bad_times,
        duration = 1,
        description = description
    )

    return bad_annots


def get_times_from_notes(raw, str_beg, str_end):
    # 
    times = raw.annotations[np.where(raw.annotations.description == str_beg)[0]].onset
    times = np.stack([times, raw.annotations[np.where(raw.annotations.description == str_end)[0]].onset], axis = 1)

    # Make sure start times are always before end times
    assert(all(times[:,0] <  times[:,1]))

    return times


def clip_iterator(raw, params, str_beg, str_end, path, fname, sfx, times = None):

    # Accept timing arrays with rows of (start,stop) or generate them
    if times == None:
        times = get_times_from_notes(raw, str_beg, str_end)

    # Clip out trials
    if params.do_clip_trials:
        for i in range(len(times[:,0])):
            new = raw.copy()
            new.crop(times[i,0], times[i,1])
            save_raw_if(new, params, path, fname + '_' + sfx + '_' + f'{i:02}')

            # Save downsampled versions
            if params.do_downsample_trials:
                new = new.resample(params.sample_freq_new)
                save_raw_if(new, params, path, fname + '_ds_' + sfx + '_' + f'{i:02}')
