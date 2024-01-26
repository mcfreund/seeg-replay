import os
import re
import mne
import numpy as np
import pandas as pd
import pickle
import bz2
import _pickle
import meegkit

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



def _cluster_contacts(signal_list):
    """ return electrode-contact hierarchy """

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


def zapline_clean(
        raw,
        nremove = 1, nfft = 1024, nkeep = None, blocksize = None, show = False
        ):
    ## from: https://mne.discourse.group/t/clean-line-noise-zapline-method-function-for-mne-using-meegkit-toolbox/7407/2
    ## uses: https://github.com/nbara/python-meegkit/blob/9204cfd8d596be479ddae932108445b4f560010a/meegkit/dss.py#L149

    data = raw.get_data().T
   
    out, artif = meegkit.dss.dss_line(
        X = data, 
        fline = raw.info["line_freq"],
        sfreq = raw.info['sfreq'],
        nremove = nremove,
        nfft = nfft,
        nkeep = nkeep,
        blocksize = blocksize,
        show = show)

    cleaned_raw = mne.io.RawArray(out.T, raw.info)
    artif_raw = mne.io.RawArray(artif.T, raw.info)

    return cleaned_raw, artif_raw



## scratch

# def load_events(
#     subject, date, session,
#     dir_orig_data = '/oscar/data/brainstorm-ws/seeg_data/Memory Task Data/Epilepsy/Monitoring/'
#     ):
#     """ loads all SEEG data from single session"""

#     dir_orig_data_sess = os.path.join(dir_orig_data, date + "_" + subject + "_" + session)
#     events_file = [os.path.join(dir_orig_data_sess, file) for file in os.listdir(dir_orig_data_sess) if 'Events.pbz2' in file]
#     if len(events_file) > 1:
#         raise Exception("Multiple event files found. " + events_file)
#     events_data = load_pkl(events_file[0])
    
#     return events_data

# # ## create stimulus channel for events:
# # n_times = signals.shape[1]
# # stim_data = np.zeros((1, n_times))
# # for samp_i in range(n_times):
# #     is_evt = events[:, 1] == samp_i
# #     if np.sum(is_evt) == 1:
# #         evt_i = np.where(is_evt)[0]
# #         stim_data[0, samp_i] = events[evt_i, 0][0]
# #     elif np.sum(is_evt) > 1:
# #         raise Exception("multiple events during same sample ... issue?")

# ## metadata:
# n_channels = len(data) + 1 ## SEEG plus one stimulus channel
# ch_names = chinfo["ch_name"].tolist() + ["stimuli"]
# #ch_names = chinfo.loc[~chinfo["is_missing"], "ch_name"].tolist() + ["stimuli"]
# ch_types = ["seeg"] * (n_channels - 1) + ["stim"]
# info = mne.create_info(ch_names, ch_types = ch_types, sfreq = sfreq)
# info["line_freq"] = line_freq
