dir_data = "/oscar/data/brainstorm-ws/megagroup_data"
dir_orig_data = '/oscar/data/brainstorm-ws/seeg_data/Memory Task Data/Epilepsy/Monitoring/'
dir_chinfo = '/oscar/data/brainstorm-ws/seeg_data/Memory Task Data/Imaging/Epilepsy'
line_freq = 60
sfreq = 1024
scale = 1e6
trigs = {
    9: "trial_start",
    20: "fix_start",
    22: "fix_resp",
    72: "clip_start",
    74: "clip_stop",
    82: "clipcue_start",
    84: "clipcue_stop",
    36: "loc_start",
    38: "loc_resp",
    56: "col_start",
    58: "col_resp",
    18: "trial_stop",
    666: "BAD_outlier"
}
inv_trigs = {v: k for k, v in trigs.items()}
epoch_info = dict(
    clip = dict(
        tmin = -1,  ## epoch start time (s), rel. row_events
        tmax = 6,  ## stop time
        metadata_tmin = -1,  ## how far BACK (s) to look for co-localized triggers/events (cols in metadata)?
        metadata_tmax = 360,
        row_events = "clip_start"  ## which event/trigger to use to define the start of a new row/epoch?
    )
)
l_freq, h_freq = 0.5, 200  ## for bandpass