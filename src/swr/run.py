from src.shared.presets    import *
from src.shared.utils      import *
from src.swr.functions_ca1 import *
from src.swr.functions_swr import *

#  Analysis pipeline:
# - collect CA1 recordings
# - bandpass filter them to 80-100 Hz and run ripple detection
# - get the ripple start and end times, plus their profiles

read     = False
paths    = PathPresets('dan-xps-15')
subjs    = ['e0010GP', 'e0011XQ', 'e0013LW', 'e0015TJ', 'e0016YR', 'e0017MC']
sessions = ['Encoding','SameDayRecall','NextDayRecall']

# Get all raw data from CA1 and assoc. channels
lfps = dill_read('./data/ca1_lfps.pt') if read else get_ca1_data(paths, sessions, subjs)

# Find SWR event times in this data
swrs = dill_read('./data/ca1_swrs.pt') if read else get_swrs_from_lfps(lfps) 

# Read all trial time info
swrs = append_swr_trial_assoc(swrs, paths, subjs, sessions)

# Plot SWRs vs everything
aggregate_and_plot_swrs_by_performance(swrs)