import os
import mne
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

from src.shared.paths import PathPresets
from src.preproc.params import ParamsPreProc
from src.preproc.utils import read_raw, read_chinfo, read_swr, swr_to_annot, \
    epoch_decim, epoch_swrs, trim_epochs, separate_channel_freq
from src.timefreq.utils import load_timefreq_h5, plot_time_frequency_heatmap

paths = PathPresets("oscar")
params = ParamsPreProc()

## Load the data

subject = 'e0010GP'
session = 'Encoding'
#region = "frontal"
params.change_pars(suffix_preproc = "no60hz_morletfull_frontal_raw")

chinfo = read_chinfo(subject, paths)


## SWRs ----

suffix_swr = "_Encoding_SameDayRecall_CMHIP2_swrs-fp-check"
swrs = read_swr(subject, suffix_swr, paths)

## from raw data, extract epochs of SWRs
raw = read_raw(subject, session, params, paths)
raw.set_annotations(swr_to_annot(swrs))

## epoch_swrs

epoch_align_event = ["SWR_beg", "SWR_mid", "SWR_end"]
epoch_baseline = [(-0.5, -0.25), (-1, 1), (0.25, 0.5)]
epoch_tmin = [-0.5, -1, -1.5]
epoch_tmax = [1.5, 1, 0.5]
plot_pars = zip(epoch_align_event, epoch_baseline, epoch_tmin, epoch_tmax)
for align, baseline_, tmin, tmax in plot_pars:
    baselines = [baseline_, None]  ## also run without baseline
    for baseline in baselines:
        params.change_pars(
            epoch_tmin = tmin,
            epoch_tmax = tmax,
            epoch_align_event = align,
            epoch_baseline = baseline)
        epochs = epoch_swrs(raw, params)
        data, times, ch_names, freqs = separate_channel_freq(epochs)
        evoked = data.mean(axis = (0))  ## average over SWRs
        plt.figure(figsize=(20, 12))
        n_row = int(np.ceil(np.sqrt(len(ch_names))))
        for i, ch in enumerate(ch_names):
            plot_time_frequency_heatmap(
                evoked[:, i, :], freqs, times, f"Channel {ch}",
                show = False, grid_size = (n_row, n_row), subplot_index = i
                )
        plt.tight_layout()
        if baseline is None:
            baseline_str = "nobaseline"
        else:
            baseline_str = "baseline"
        plt.savefig(os.path.join("figs", "timefreq", f"evoked_frontal_{baseline_str}_{align}.png"))
        plt.show()
        plt.close()

