# This class holds all control parameters for pre-processing
class ParamsPreProc:
    def __init__(s):

        # What should the pre-processing script do?
        s.do_events   = False     # Write events files?
        s.do_sessinfo = False     # Write session info?

        s.do_bandpass = True      # Bandpass filtering
        s.do_epoching = False     # Epoching
        s.do_rerefing = False     # Re-referencing
        s.do_rmline   = True      # Line-noise removal
        s.do_rmouts   = True      # Outlier removal via MAD threshold
        s.do_interact = False     # Interactive MNE outlier removal
        s.do_apply_bads = True    # Apply bad channel definitions (see below)
        s.do_apply_n_components_60hz = True # Apply non-default exclusions for attenuating line noise 

        # Dictionary of bad channels that is optionally applied to raw (do_apply_bads):
        s.bads = {
            "e0010GP_Encoding":         ["E-PFC9"],
            "e0010GP_SameDayRecall":    ["E-PFC9"],
            "e0010GP_NextDayRecall":    ["E-PFC9"],
            "e0017MC_Encoding":         ["L-HIPB11"],
            "e0017MC_NextDayRecall":    ["L-HIPB11"],
            "e0017MC_SameDayRecall":    ["L-HIPB11"]
        }

        # Dictionary of number of spatial components to remove for attenuating line noise:
        # (only applied if non-default)
        s.n_components_60hz = dict(
            e0015TJ_Encoding      = 5,
            e0015TJ_SameDayRecall = 5,
            e0015TJ_NextDayRecall = 5,
            e0016YR_Encoding      = 5,
            e0016YR_SameDayRecall = 5
        )

        # Downsample in various places
        s.do_downsample_epochs  = True # Doesn't exist yet
        s.do_downsample_session = True
        s.do_downsample_trials  = True
        
        # Removes non-experiment time from beg, end of main raw file.
        s.do_trim_pre  = False
        s.do_trim_post = False

        # Clip sections into files (conflicts w/ trim)
        s.do_clip_pre    = True
        s.do_clip_dur    = True
        s.do_clip_post   = True
        s.do_clip_trials = True

        # Which formats should data be exported in
        s.save_fif = True
        s.save_set = False
        s.save_h5  = False
        s.save_csv = False
        s.save_plt = False

        # Continuous behavioral data


        # Save files for intermediate preprocessing steps?
        s.save_step_rerefing   = False
        s.save_step_epoching   = False
        s.save_step_bandpass   = False
        s.save_step_rmline     = False
        s.save_step_rmout      = False

        # Should we use sub-directories or save to one folder?
        s.use_subdirs = True
        
        # Which subjects to process (leave empty for all)
        s.subj_list = ['e0011XQ']

        # Prefix for all data operations (read, write)
        s.path_base = '/oscar/data/brainstorm-ws'
        s.path_figs = s.path_base + '/megagroup_data/figs'
        s.path_read = s.path_base + '/seeg_data/Memory Task Data/Epilepsy/Monitoring/'
        #s.path_save = s.path_base + '/megagroup_data'
        s.path_save = s.path_base + '/tmp'
        s.path_chnl = '/oscar/data/brainstorm-ws/seeg_data/Memory Task Data/Imaging/Epilepsy'

        # s.path_base = '/home/dan/projects/work/seeg_data'
        # s.path_figs = '/home/dan/projects/work/seeg_data/figs'
        # s.path_read = s.path_base + '/original'
        # s.path_chnl = s.path_base + '/original/Imaging/Epilepsy'
        # s.path_save = s.path_base + '/processed'

        # How many parallel raw-file creation jobs?
        s.n_jobs = 1

        # Line noise frequency
        s.line_freq = 60

        # Referencing method
        s.reref_method = 'laplacian'

        # Initial frequency
        s.sample_freq_native = 1024

        # New sampling frequency
        s.sample_freq_new = 1024/4

        # Factor to rescale data by
        s.scale = 1e6

        # Define trigger meanings
        s.trigger_to_desc_dict = {
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

        # Reverse mapping for triggers
        s.trigger_from_desc_dict = {v: k for k, v in s.trigger_to_desc_dict.items()}

        # Epoch definitions
        s.epoch_info = dict(
            clip = dict(
                tmin = -1,  ## epoch start time (s), rel. row_events
                tmax = 6,   ## stop time
                metadata_tmin = -1,  ## how far BACK (s) to look for co-localized triggers/events (cols in metadata)?
                metadata_tmax = 360,
                row_events = "clip_start"  ## which event/trigger to use to define the start of a new row/epoch?
            )
        )
                
        # Bandpass (low, high frequencies)
        s.bandpass = (0.5, 200)
        
        # Filename suffix: for reading pre-processed raws, and writing derivatives of them (e.g., epochs)
        # NB: does not include the trailing suffix and .fif extension that mne expects (e.g., '-epo.fif', 'raw.fif')
        s.suffix_preproc = "_no60hz_bp_rmouts"
        
