import os
import re
import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.preproc.constants import dir_data, inv_trigs, epoch_info

do_mark_bads = True

session_info = pd.read_csv(os.path.join(dir_data, "session_info.csv"))
beh_data = pd.read_csv(os.path.join(dir_data, "behavioral_data.csv"))
inv_trigs.update({"BAD_outlier": 666})  ## add code for outlier

for i, d in session_info.iterrows():
    if i == 10: break
    print("Processing subject " + d["participant_id"] + ", session " + d["session"] + "...")
    print("File " + str(i + 1) + " of " + str(len(session_info)))

    ## get paths, fnames:
    dir_subj = os.path.join(dir_data, d["participant_id"])
    dir_sess = os.path.join(dir_subj, d["session"])
    dir_out = os.path.join(dir_data, "epochs", d["participant_id"], d["session"])
    os.makedirs(dir_out, exist_ok = True)
    fname_base = d["participant_id"] + "_" + d["session"]

    raw = mne.io.read_raw_fif(os.path.join(dir_sess, fname_base + "_no60hz_ref_bp_raw.fif"), preload = True)
    chinfo = pd.read_csv(os.path.join(dir_subj, d["participant_id"] + "_chinfo.csv"))
    events, event_id = mne.events_from_annotations(raw, event_id = inv_trigs)
    print(raw.annotations)

    for epoch_type in epoch_info.keys():
        
        ## create metadata df from events/trigs:
        metadata, events_out, event_id_out = mne.epochs.make_metadata(
            events = events,
            event_id = event_id,
            tmin = epoch_info[epoch_type]["metadata_tmin"],
            tmax = epoch_info[epoch_type]["metadata_tmax"],
            sfreq = raw.info["sfreq"],
            row_events = epoch_info[epoch_type]["row_events"]
        )

        ## subset beh_data to this subject and session:
        beh_data_sub = beh_data.loc[(beh_data["participant_id"] == d["participant_id"]) & (beh_data["session"] == d["session"]), :]

        ## bind columns of beh_data_sub and metadata, only if there are the same number of rows (otherwise, raise error)
        if beh_data_sub.shape[0] != metadata.shape[0]:
            raise ValueError("Number of rows in metadata and beh_data_sub do not match!")
        metadata.reset_index(drop = True, inplace = True)
        beh_data_sub.reset_index(drop = True, inplace = True)
        metadata_new = pd.concat([metadata, beh_data_sub], axis = 1)
        if metadata_new.shape[0] != metadata.shape[0]:
            raise ValueError("Number of rows in metadata_new and metadata do not match!")
        
        ## epoch and save
        epochs = mne.Epochs(
            raw,
            baseline = None,
            detrend = None,
            events = events_out,
            event_id = event_id_out,
            tmin = epoch_info[epoch_type]["tmin"],
            tmax = epoch_info[epoch_type]["tmax"],
            preload = True,
            metadata = metadata_new,
            reject_by_annotation = False
            )
        ## save for mne:
        fname_epochs = os.path.join(dir_out, fname_base + "_no60hz_ref_bp_" + epoch_type + "-epo.fif")
        epochs.save(fname_epochs, overwrite = True)

        ## export to eeglab, csv:
        epochs.export(fname_epochs.replace("fif", "eeglab"), overwrite = True)
        epochs.to_data_frame().to_csv(fname_epochs.replace("fif", "csv"))
        epochs.metadata.to_csv(fname_epochs.replace("fif", "_metadata.csv"))
