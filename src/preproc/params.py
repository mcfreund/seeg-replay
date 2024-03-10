# This class holds all control parameters for pre-processing
class ParamsPreProc:
    def __init__(s):

        import numpy as np
        from src.preproc.utils import tf_binning_matrix

        # What should the pre-processing script do?
        s.do_events    = True     # Write events files?
        s.do_sessinfo  = True     # Write session info?
        s.do_make_raws = False

        s.do_bandpass = False      # Bandpass filtering
        s.do_epoching = False     # Epoching
        s.do_rerefing = True      # Re-referencing
        s.do_rmline   = True      # Line-noise removal
        s.do_rmouts   = False     # Outlier removal via MAD threshold
        s.do_interact = False     # Interactive MNE outlier removal
        s.do_apply_bads = True    # Apply bad channel definitions (see below)
        s.do_apply_n_components_60hz = True # Apply non-default exclusions for attenuating line noise 

        # Dictionary of bad channels that is optionally applied to raw (do_apply_bads):
        s.bads = {
            "e0010GP_Encoding":         ["E-PFC7"],
            "e0010GP_SameDayRecall":    ["E-PFC7"],
            "e0010GP_NextDayRecall":    ["E-PFC7"],
            "e0017MC_Encoding":         ["R-PINS9"],
            "e0017MC_NextDayRecall":    ["R-PINS9"],  ## only bad in this session, but exclude from all for simplicity
            "e0017MC_SameDayRecall":    ["R-PINS9"],
            # "e0015TJ_Encoding":         ["R-AMY1"],  ## looked odd, but not extreme as others subjs' bads.
            # "e0015TJ_NextDayRecall":    ["R-AMY1"],
            # "e0015TJ_SameDayRecall":    ["R-AMY1"]
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
        s.do_downsample_epochs  = True
        s.do_downsample_session = False
        s.do_downsample_trials  = True
        
        # Removes non-experiment time from beg, end of main raw file.
        s.do_trim_pre  = False
        s.do_trim_post = False

        # Clip sections into files (conflicts w/ trim)
        s.do_clip_pre    = False
        s.do_clip_dur    = False
        s.do_clip_post   = False
        s.do_clip_trials = False

        # Which formats should data be exported in
        s.save_fif = True
        s.save_set = False
        s.save_h5  = False
        s.save_csv = False
        s.save_plt = False

        # Save files for intermediate preprocessing steps?
        s.save_step_rerefing   = True
        s.save_step_epoching   = False
        s.save_step_bandpass   = True
        s.save_step_rmline     = True
        s.save_step_rmout      = False

        # Should we use sub-directories or save to one folder?
        s.use_subdirs = True
        
        # Which subjects to process (leave empty for all)
        s.subj_list = []

        # How many parallel raw-file creation jobs?
        s.n_jobs = 4

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
            ),
            clipcue = dict(
                tmin = -1,
                tmax = 6,
                metadata_tmin = -1,
                metadata_tmax = 360,
                row_events = "clipcue_start"
            ),
            loc = dict(
                tmin = -1,
                tmax = 6,
                metadata_tmin = -1,
                metadata_tmax = 360,
                row_events = "loc_start"
            ),
            col = dict(
                tmin = -1,
                tmax = 6,
                metadata_tmin = -1,
                metadata_tmax = 360,
                row_events = "col_start"
            )
        )
                
        # Bandpass (low, high frequencies)
        s.bandpass = (0.5, 200)
        
        # Filename suffix: for reading pre-processed raws, and writing derivatives of them (e.g., epochs)
        # NB: does not include the trailing suffix and .fif extension that mne expects (e.g., '-epo.fif', 'raw.fif')
        s.suffix_preproc = "_no60hz_bp"

        # Time-frequency decomposition parameters        
        
        s.tf_max_freq = 150
        s.tf_min_freq = 1
        s.tf_step = 2
        ## peak wavelet frequencies:
        ## (could also do log spacing, but this would overrepresent low vs high frequencies)
        s.tf_freqs = np.arange(s.tf_min_freq, s.tf_max_freq + 1, s.tf_step)
        s.tf_n_freq = len(s.tf_freqs)
        #s.tf_freqs = np.linspace(s.tf_min_freq, s.tf_max_freq, num = s.tf_n_freq)
        s.tf_min_n_cycles = 3  ## controls width of gaussians in time domain
        s.tf_max_n_cycles = 20
        s.tf_n_cycles = np.linspace(s.tf_min_n_cycles, s.tf_max_n_cycles, num = s.tf_n_freq)
        s.tf_n_jobs = 32
        ## regions to subset for time-frequency files (just to keep file sizes manageable)
        ## 
        s.tf_regions = ["frontal", "temporal", "hippocampus|amygdala"]
        ## their corresponding filename suffix strings:
        s.tf_regions_fnames = ["frontal", "temporal", "hcamy"]
        ## bands:
        s.tf_bands = dict(
            delta = [1, 3],
            theta = [4, 7],
            alpha = [7, 12],
            beta = [13, 31],
            lowgamma = [31, 70],
            highgamma = [71, 150]
        )
        s.tf_binning_mat = tf_binning_matrix(s.tf_freqs, s.tf_bands)


    def change_pars(s, **kwargs):
        for key, value in kwargs.items():
            setattr(s, key, value)