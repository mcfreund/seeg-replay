from src.shared.paths      import *
from src.shared.utils      import *
from src.swr.functions_ca1 import *
from src.swr.functions_swr import *
import ipdb

read     = False
paths    = PathPresets('dan-xps-15')
subjs    = ['e0010GP']#, 'e0011XQ', 'e0013LW', 'e0015TJ', 'e0016YR', 'e0017MC']
sessions = ['Encoding']#,'SameDayRecall','NextDayRecall']

# Get raw data from CA1 and assoc. channels, process to SWRs
lfps = dill_read('./data/ca1_lfps.pt') if read else get_lfp_data(paths, sessions, subjs, save_pkl=False)
swrs = dill_read('./data/ca1_swrs.pt') if read else get_swrs_from_lfps(lfps, save_pkl=False, save_swrs=False)

# Get raw data from CA1 and assoc. channels, process to SWRs
#get_swr_bp_lfp_power_z(paths, subjs, sessions)
#swrs = dill_read('./data/ca1_swrs.pt') if read else get_swrs_from_raws(paths, subjs, sessions, save_pkl=True, save_swrs=True, inspect_swrs=False)

# Read all trial time info
swrs = append_swr_trial_assoc(swrs, paths, subjs, sessions)

# Plot SWRs vs everything
df = aggregate_swrs_and_behavior(swrs)