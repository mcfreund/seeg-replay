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
    18: "trial_stop"
}
good_subjects = ["e0010GP", "e0011XQ", "e0015TJ", "e0016YR", "e0017MC", "e0018RI"]
