## read in raw SEEG timeseries per subject*session and save in single file.
## bring chinfo in register with SEEG data and save as corresponding csv.
## for functions in src to be imported as below, run from root directory of repo.

import os
import re
import mne
import numpy as np
import pandas as pd
import pickle
import bz2
import _pickle
from joblib import Parallel, delayed

from src.preproc.constants import dir_data
from src.preproc.utils import construct_raw

session_info = pd.read_csv(os.path.join(dir_data, "session_info.csv"))


## construct all raw timeseries files and save in megagroup_data directory:

def read_raw(i, d):
    """ read raw timeseries for a single session. also reads metadata about channels, in chinfo. """
    
    raw, chinfo = construct_raw(subject = d["participant_id"], session = d["session"], subdir_orig = d["subdir_orig"])
    
    dir_subj = os.path.join(dir_data, d["participant_id"])
    fname_base = d["participant_id"] + "_" + d["session"]

    raw.save(os.path.join(dir_subj, d["session"], fname_base + "_raw.fif"))
    mne.export.export_raw(os.path.join(dir_subj, d["session"], fname_base + "_raw.vhdr"), raw)  ## alt format
    chinfo.to_csv(os.path.join(dir_subj, fname_base + "_chinfo.csv"))
    
    return raw, chinfo


raws = Parallel(n_jobs = 8)(delayed(read_raw)(i, d) for i, d in session_info.iterrows())
