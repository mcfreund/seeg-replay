import os
import re
import mne
import pickle
import meegkit
import scipy


import pandas as pd
import scipy.io as sio
import numpy as np
import itertools
import bz2
import _pickle

def load_pkl(path):
    """ loads compressed pickle file """

    with bz2.open(path, 'rb') as f:
        neural_data = _pickle.load(f)
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

def read_subj_channel_info(subj, path_chnl):
    """ wrangles channel / anatomical info from excel files. optionally sorts row order to match channel order in a signal_labels object."""

    # Read channel information file
    chinfo = pd.read_excel(os.path.join(path_chnl, subj, "parsed.xlsx"))

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

    return data, contacts

def construct_raw(session_row, params):
    """ Constructs raw data object for a given day. Also loads metadata about channels, in chinfo. 
    NB: only the intersection of channel names in chinfo["channel"] and names of the SEEG day subdirectory is kept.
    I.e., we keep only channels for which we have both data and their location (assuming no typos in names).
    """
    subj = session_row['participant_id']
    sess = session_row['session']
    dir  = session_row['subdir_orig']

    # Read channel information
    chinfo = read_subj_channel_info(subj, params.path_chnl)

    # Read SEEG data in intersection of contacts data files
    data, contacts = load_session_seeg(dir = params.path_read + '/' + dir, contacts = chinfo["contact"])

    # Keep only contacts from data in chinfo
    chinfo = chinfo[chinfo["contact"].isin(contacts)]
    
    # Read events file (from processed data path)
    events = pd.read_csv(os.path.join(session_row['path_sess'], subj + "_" + sess + "_events.csv"))
    
    # Insert column of zeros in events (why?)
    events = np.stack([events["time"], np.zeros(len(events)), events["code"]]).T.astype(int)
    
    # Rescale signal
    signals = np.stack([d["signal"] for d in data]) / params.scale # ideally, in V

    # MNE metadata
    n_channels = len(data)
    ch_names   = chinfo["contact"].tolist()
    ch_types   = ["seeg"] * n_channels

    # Create MNE structure
    info = mne.create_info(ch_names, ch_types = ch_types, sfreq = params.sample_freq_native)

    # Add line frequency to MNE structure
    info["line_freq"] = params.line_freq

    ## Construct MNE "raw" type (ensure signals and stim data order match ch_types/names)
    raw = mne.io.RawArray(signals, info)
    
    ## Add events as annotations:
    annots = mne.annotations_from_events(events, params.sample_freq_native, event_desc = params.trigger_to_desc_dict)
    raw.set_annotations(annots)
    
    return raw, chinfo


# Save selector for raws
def save_raw_if(raw, params, path_sess, fname):
    ''' Function for saving file types selected in params.'''

    # Filename without a filetype suffix
    fname = os.path.join(path_sess, fname)

    # Save .fif format
    if params.save_fif:
        raw.save(fname + '.fif', overwrite = True)

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
def save_plt_if(raw, params, fname):
    ''' Function for saving plots, if selected in params.'''

    if params.save_plt:
        fig = raw.compute_psd().plot(show = False)
        fig.axes[0].set_title("raw timeseries: " + fname + "\n" + t)
        fn_psd = os.path.join(params.path_figs, fname ,'.png')
        fig.savefig(fn_psd)



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



# def load_cpkl(path):
#     """ loads compressed pickle file called by load_electrodes() """

#     with bz2.open(path, 'rb') as f:
#         neural_data = cPickle.load(f)
#     return neural_data



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

    return raw_out


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


def rereference(raw, chinfo, method = 'laplacian', drop_bads = True, selfref_first = False):
    """ wrapper for feeding mne objects into  _ref_electrodes() """

    raw_copy = raw.copy()
    if drop_bads:
        raw_copy.drop_channels(raw_copy.info["bads"])
    
    data_ref, ch_names_ref =_reref_electrodes(
        signals = raw_copy.get_data(),
        signal_list = raw_copy.ch_names,
        method = method,
        selfref_first = selfref_first)

    ## mark chinfo with those that were rereferenced
    chinfo["was_rereferenced"] = chinfo["contact"].isin(ch_names_ref)

    ## add data back to raw
    dropped_chs = [ch for ch in raw_copy.ch_names if ch not in ch_names_ref]  ## drop
    raw_copy.drop_channels(dropped_chs)
    if raw_copy.ch_names != ch_names_ref:
        raise Exception("ch_names_ref and raw_copy.ch_names do not match.")
    raw_copy._data = data_ref

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

def clip_iterator(raw, params, str_beg, str_end, path, fname, sfx):

    # Get array with rows of (start, stop)
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
