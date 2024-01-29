## This script is used to clean raw data and save the cleaned data to disk.
## It also marks bad channels and bad time samples, and saves the bad channel info (_chinfo).
## The cleaned data is saved in both .fif and .set formats, for use in MNE and EEGLAB, respectively.
## The raw data is cleaned by marking bad channels, attenuating power-line noise, re-referencing, and bandpass filtering.
## The cleaned data is also annotated with bad time samples, which are detected using a simple thresholding method.
## Power-line noise is attenuated via zapline, which defines a spatial filter for powerline noise and removes it 
## from the data. This requires specifying the number of (spatial) dimensions the noise spans. Default is 3.
## If line-noise persists (see figs), you can increase the number of dimensions by editing the value in the
## session_info.csv file.
## Marking bad channels is done manually, by plotting the raw data and marking bad channels interactively via
## mne popup window (best done via Qt backend). When a raw plot file is open, you can mark bad channels by
## clicking on the channel trace or name in the plot window. CLose the plot to let the script proceed. 
## The bad channels are then saved to the _chinfo file and the raw file.
## NB: The best way to use mne's interactive graphical windows on oscar is (unfortunately) via virtual desktop
##     on OOD. Displaying larger files in the popup window can be slow -- potentially plotting onnly a subset
##     of timepoints or channels can speed this up.

import os
import re
import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.preproc.utils import zapline_clean, rereference, annot_bad_times
from src.preproc.constants import dir_data, l_freq, h_freq

do_mark_bads = False
rerun_subj = ["e0010GP", "e0011XQ", "e0013LW", "e0015TJ", "e0016YR", "e0017MC"]

session_info = pd.read_csv(os.path.join(dir_data, "session_info.csv"))


## first mark bad channels for all files:

if do_mark_bads:
    for i, d in session_info.iterrows():
        print("Marking bads for subject " + d["participant_id"] + ", session " + d["session"] + "...")
        print("File " + str(i + 1) + " of " + str(len(session_info)))

        if not d["participant_id"] in rerun_subj:
            print("Skipping this subject.")
            continue

        ## get paths:
        dir_subj = os.path.join(dir_data, d["participant_id"])
        dir_sess = os.path.join(dir_subj, d["session"])
        fname_base = d["participant_id"] + "_" + d["session"]

        raw = mne.io.read_raw_fif(os.path.join(dir_sess, fname_base + "_raw.fif"), preload = True)
        chinfo = pd.read_csv(os.path.join(dir_subj, d["participant_id"] + "_chinfo.csv"))

        ## manually mark bad channels and update each subject's chinfo
        raw.compute_psd().plot()
        raw.plot(block = True)  ## halts loop until plot is closed; use to mark bads.
        is_bad = [ch in raw.info["bads"] for ch in chinfo["contact"]]
        chinfo["is_bad_sess_" + d["session"]] = is_bad

        ## save/update
        chinfo.to_csv(os.path.join(dir_subj, d["participant_id"] + "_chinfo.csv"), index = False)
        raw.save(os.path.join(dir_sess, fname_base + "_raw.fif"), overwrite = True)
    
    print("done marking bads.")

## proceed with rest of preprocessing:

for i, d in session_info.iterrows():
    print("Processing subject " + d["participant_id"] + ", session " + d["session"] + "...")
    print("File " + str(i + 1) + " of " + str(len(session_info)))

    if not d["participant_id"] in rerun_subj:
        print("Skipping this subject.")
        continue

    ## get paths:
    dir_subj = os.path.join(dir_data, d["participant_id"])
    dir_sess = os.path.join(dir_subj, d["session"])
    fname_base = d["participant_id"] + "_" + d["session"]

    raw = mne.io.read_raw_fif(os.path.join(dir_sess, fname_base + "_raw.fif"), preload = True)
    chinfo = pd.read_csv(os.path.join(dir_subj, d["participant_id"] + "_chinfo.csv"))

    ## attenuate power-line noise
    ## check if "n_remove_60hz" is in d, which is a series pandas object:
    if "n_remove_60hz" in d.keys():
        n_remove = d["n_remove_60hz"]
    else:
        n_remove = 3
    raw_no60hz, raw_60hz = zapline_clean(raw, nremove = n_remove)  ## NB: this does not exclude bad chs

    ## re-reference (laplacian)
    
    raw_no60hz_ref, chinfo = rereference(
        raw_no60hz, chinfo,
        method = "laplacian",
        drop_bads = True,
        selfref_first = False)

    ## filter
    raw_no60hz_ref_bp = raw_no60hz_ref.copy().filter(l_freq, h_freq)

    ## annotate bad time samples
    
    bad_annots = annot_bad_times(raw_no60hz_ref_bp, thresh = 20, consensus = 1, method = "mad", duration = 6/1024)
    raw_no60hz_ref_bp.set_annotations(raw_no60hz_ref_bp.annotations + bad_annots)

    ## save/update
    
    chinfo.to_csv(os.path.join(dir_subj, d["participant_id"] + "_chinfo.csv"), index = False)
    raw_no60hz.save(os.path.join(dir_sess, fname_base + "_no60hz_raw.fif"), overwrite = True)
    raw_no60hz_ref.save(os.path.join(dir_sess, fname_base + "_no60hz_ref_raw.fif"), overwrite = True)
    raw_no60hz_ref_bp.save(os.path.join(dir_sess, fname_base + "_no60hz_ref_bp_raw.fif"), overwrite = True)

    ## export to eeglab:
    mne.export.export_raw(os.path.join(dir_sess, fname_base + "_raw.set"), raw, overwrite = True)
    mne.export.export_raw(os.path.join(dir_sess, fname_base + "_no60hz_ref_bp_raw.set"), raw_no60hz_ref_bp, overwrite = True)

    ## save plots:
    raws = [raw, raw_no60hz, raw_no60hz_ref, raw_no60hz_ref_bp]
    titles = [
        "raw",
        "n_remove_60hz = " + str(n_remove),
        "n_remove_60hz = " + str(n_remove) + " ref = laplacian",
        "n_remove_60hz = " + str(n_remove) + " ref = laplacian, bandpass = " + str(l_freq) + "-" + str(h_freq)
        ]
    for r, t in zip(raws, titles):
        ## psds:
        fig = r.compute_psd().plot(show = False)
        fig.axes[0].set_title("raw timeseries: " + fname_base + "\n" + t)
        fn_psd = os.path.join(os.path.abspath("figs"), "raw", fname_base + "_" + t + "_psd.png")
        fig.savefig(fn_psd)
