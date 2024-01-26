import os
import re
import mne
import numpy as np
import pandas as pd
import pickle
import bz2
import _pickle
from joblib import Parallel, delayed

from src.preproc.utils import zapline_clean
from src.preproc.constants import trigs, dir_data, dir_orig_data
from src.preproc.utils import construct_raw


## TODO:
## preprocessing
## 1. mark bad channels
## 2. attenuate power-line noise
## 3. re-reference
## 4. annotate bad time samples
## 5. filter
## 6. epoch (use beh data files)


## manually mark bad channels and update each subject's chinfo

if 'bad_chs' in locals() or 'bad_chs' in globals():
    print('ned to implement')
    for (raw, chinfo), (i, d) in zip(raws, session_info.iterrows()):
       is_bad = [ch in bad_chs for ch in chinfo["ch_name"]]
       chinfo["is_bad_sess-" + d["session"]] = is_bad
else:
    for (raw, chinfo), (i, d) in zip(raws, session_info.iterrows()):
        raw.compute_psd().plot()
        raw.plot(block = True)  ## halts loop until plot is closed (i.e., after bad marked); NB: modifies data in raws
        is_bad = [ch in raw.info["bads"] for ch in chinfo["ch_name"]]
        chinfo["is_bad_sess-" + d["session"]] = is_bad  ##  NB: modifies data in raws


## attenuate power-line noise

raws_no60hz, raws_60hz = [], []
for raw, chinfo in raws:
    raw_clean, raw_artif = zapline_clean(raw)
    raws_no60hz.append(raw_clean)
    raws_60hz.append(raw_artif)

#raw_clean, raw_artif = zapline_clean(raw)
#raw.compute_psd().plot()
#raw_clean.compute_psd().plot()
#raw_artif.compute_psd().plot()
#raw.plot()
#raw_clean.plot()
#raw_artif.plot()


## re-reference



## annotate bad time samples



## filter




## scratch ---
    #raws[i][1] = chinfo



#raws[0][0].plot()


##"e0015TJ"  ## something's up with this subject's channels.


## epoching ---



## scratch 
    # session_info = []
# for item in os.listdir(dir_data):
#     if os.path.isdir(os.path.join(dir_data, item)):
#         date, subject, session = item.split('_')
#         session_info.append([date, subject, session])
# session_info = pd.DataFrame(session_info, columns = ['date', 'subject', 'session'])
# #session_info = session_info[session_info["subject"].isin(good_subjects)]
# session_info = session_info.reset_index()

# subject = subject_info.iloc[0]["participant_id"]
# session = subject_info.iloc[0]["session"]
# subdir_orig = subject_info.iloc[0]["subdir_orig"]
# raw, chinfo = construct_raw(subject, session, subdir_orig)
