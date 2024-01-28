import os
import re
import mne
import numpy as np
import pandas as pd
import pickle
import bz2
import _pickle
import meegkit
import scipy

from src.preproc.constants import trigs, dir_data, dir_orig_data, dir_chinfo, sfreq, scale, line_freq


def load_pkl(path):
    """ loads compressed pickle file """

    with bz2.open(path, 'rb') as f:
        neural_data = _pickle.load(f)
    return neural_data


def load_session(
    subdir,
    contacts = None,
    dir_seeg = dir_orig_data
    ):
    """ loads all SEEG data from single session"""

    dir_sess = os.path.join(dir_seeg, subdir)
    if contacts is None:
        ## list of all in directory:
        data_files = [os.path.join(dir_sess, file) for file in os.listdir(dir_sess) if '.pbz2' in file and "Events" not in file]
    else:
        ## list of only those specified in contacts. if specified, ensures set and order of data matches chinfo.
        ## find all files that end in c + ".pbz2" for c in contacts:
        ## get all files in filelist that match pattern in contacts:
        data_files = [os.path.join(dir_sess, file) for file in os.listdir(dir_sess) if any([c + '.pbz2' in file for c in contacts])]
        contacts = [contact for contact, file in zip(contacts, data_files) if os.path.exists(file)]
        
    data = [load_pkl(fn) for fn in data_files]

    return data, contacts


def load_chinfo(
    subject,
    dir_chinfo = dir_chinfo
    ):
    """ wrangles channel / anatomical info from excel files. optionally sorts row order to match channel order in a signal_labels object."""

    chinfo = pd.read_excel(os.path.join(dir_chinfo, subject, "parsed.xlsx"))
    ## create new columns in chinfo that separate electrode and site/contact info:
    ## NB: sites nested in electrodes
    chinfo[["electrode", "site"]] = chinfo["contact"].str.extract('([a-zA-Z-]+)(\d+)', expand = True)
    chinfo = chinfo.sort_values("index")
    ## create col that will become ch_name in raw:
    #chinfo["ch_name"] = chinfo["Anatomical Label"] + "_wm-" + chinfo["White Matter"].astype("string")
    ## exception for formatting:
    if subject == "e0015TJ":
        chinfo["contact"] = chinfo["contact"].str.replace('-', '')
    
    return chinfo


def construct_raw(
    subject, session, subdir_orig,
    dir_data = dir_data,
    trigs = trigs,
    sfreq = sfreq,
    scale = scale,
    line_freq = line_freq
    ):
    """ Constructs raw data object for a given day. Also loads metadata about channels, in chinfo. 
    NB: only the intersection of channel names in chinfo["channel"] and names of the SEEG day subdirectory is kept.
    I.e., we keep only channels for which we have both data and their location (assuming no typos in names).
    """

    ## load data and events files:   

    chinfo = load_chinfo(subject)
    data, contacts = load_session(subdir = subdir_orig, contacts = chinfo["contact"])  ## intersection of contacts and data_files taken
    chinfo = chinfo[chinfo["contact"].isin(contacts)]  ## keep chinfo aligned with data
    
    events = pd.read_csv(os.path.join(dir_data, subject, session, subject + "_" + session + "_events.csv"))
    events = np.stack([events["time"], np.zeros(len(events)), events["code"]]).T.astype(int)
    
    signals = np.stack([d["signal"] for d in data]) / scale # ideally, in V

    n_channels = len(data) ## SEEG plus one stimulus channel
    ch_names = chinfo["contact"].tolist()
    ch_types = ["seeg"] * n_channels
    info = mne.create_info(ch_names, ch_types = ch_types, sfreq = sfreq)
    info["line_freq"] = line_freq

    ## construct raw (ensure signals and stim data order match ch_types/names)
    raw = mne.io.RawArray(signals, info)
    
    ## add events as annotations:
    annots = mne.annotations_from_events(events, sfreq, event_desc = trigs)
    raw.set_annotations(annots)
    
    return raw, chinfo


def zapline_clean(
        raw,
        nremove = 1, nfft = 1024, nkeep = None, blocksize = None, show = False
        ):
    ## adapted from: https://mne.discourse.group/t/clean-line-noise-zapline-method-function-for-mne-using-meegkit-toolbox/7407/2
    ## uses: https://github.com/nbara/python-meegkit/blob/9204cfd8d596be479ddae932108445b4f560010a/meegkit/dss.py#L149

    cleaned_raw, artif_raw = raw.copy(), raw.copy()
    out, artif = meegkit.dss.dss_line(
        X = raw.get_data().T, 
        fline = raw.info["line_freq"],
        sfreq = raw.info['sfreq'],
        nremove = nremove,
        nfft = nfft,
        nkeep = nkeep,
        blocksize = blocksize,
        show = show)

    cleaned_raw._data = out.T
    artif_raw._data = artif.T

    return cleaned_raw, artif_raw


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


def rereference(
    raw, chinfo,
    method = 'laplacian',
    drop_bads = True,
    selfref_first = False
    ):
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

