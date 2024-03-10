import mne
import scipy as sp
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

from src.shared.utils import *

def plot_neural_data(paths, subj, sess, trials):

    loc_dict, data = get_loc_dict_and_data(paths, subj, sess)

    # Look at every unique location
    for loc in loc_dict:

        Parallel(n_jobs = 4)(delayed(plot_wrap)(data, subj, sess, loc, ctct, cnum, trials) for cnum, ctct in enumerate(loc_dict[loc]))


def plot_wrap(data, subj, sess, loc, ctct, cnum, trials):
    title = f'{subj} {sess} {loc.replace(" ","-")} {cnum} {ctct}'
    plot_neural_trial(data, ctct, trials, title)


def plot_neural_trial(data, ctct, trials, title, save_dir='./figs/neural/'):

    # Turn interactive plotting off
    plt.ioff()

    fs = int(data.info['sfreq'])

    # Get all trial time and index information
    code_info = {'time':{}, 'idx':{}}
    for code in np.unique(data.annotations.description):
        match_ids = [i for i, candidate in enumerate(data.annotations.description) if candidate == code]
        code_info['time'][code] = [data.annotations[id]['onset'] for id in match_ids]
        code_info['idx'][code] = data.time_as_index(code_info['time'][code])

    # Create a plot for every trial
    for trl in trials:

        # Agument title with trial to label this plot
        label = title + f' Trial {trl:02d}'

        # Get event timing and indices into data
        idx_beg = code_info['idx']['trial_start'][trl]
        idx_end = code_info['idx']['trial_stop' ][trl]

        period_loc = (code_info['time']['loc_start'    ][trl], code_info['time']['loc_resp'    ][trl])
        period_clp = (code_info['time']['clip_start'   ][trl], code_info['time']['clip_stop'   ][trl])
        period_cue = (code_info['time']['clipcue_start'][trl], code_info['time']['clipcue_stop'][trl])
        period_col = (code_info['time']['col_start'    ][trl], code_info['time']['col_resp'    ][trl])

        # Get voltage and times
        x    = data[ctct][0].flatten()[idx_beg:idx_end]
        time = data[ctct][1].flatten()[idx_beg:idx_end]

        # Create figure
        plt.figure(figsize=[12,12])
        plt.suptitle(title + f' Trial {trl}')

        # Contact trace plot with task periods
        plt.subplot(2,1,1)
        plt.title(f'Contact trace')
        
        # Plot the electrode trace
        plt.plot(time, x)

        # Shade the task periods
        plt.axvspan(period_clp[0], period_clp[1], alpha=0.2, color='green')
        plt.axvspan(period_cue[0], period_cue[1], alpha=0.2, color='blue')
        plt.axvspan(period_loc[0], period_loc[1], alpha=0.2, color='red')
        plt.axvspan(period_col[0], period_col[1], alpha=0.2, color='yellow')

        # Label
        plt.legend(['voltage','clip','cue','loc','col'])

        # Set xlimits to actual times
        plt.xlim([time[0],time[-1]])
        plt.xlabel('Time [s]')
        plt.ylabel('Voltage')

        # Plot for power by frequency
        plt.subplot(2,1,2)
        plt.title('Sliding FFT')

        # Setup sliding FFT
        win = sp.signal.windows.gaussian(fs, std=fs/4, sym=True)
        SFT = sp.signal.ShortTimeFFT(win, hop=10, fs=fs, scale_to='magnitude', )

        # Run FFT
        Sx = SFT.stft(x)

        # Determine where the SFT is < 30 Hz
        rows = SFT.f < 40

        # Get indices into SFT data
        #padding = Sx.shape[1] - time[0::10].shape[0]
        #delta = int((padding-1)/2)
        #sinds = [(delta+1), (Sx.shape[1]-delta)]
        cols  = range(Sx.shape[1])

        # Plot it
        #plt.imshow(abs(Sx[rows,:]), origin='lower', aspect='auto', extent=SFT.extent(idx_end-idx_beg), cmap='viridis')
        plt.pcolor(cols,SFT.f[rows],abs(Sx[rows,:]))
        plt.xlabel('Samples')
        plt.ylabel('Frequency')

        # Squeeze everything
        plt.tight_layout()

        # Save
        plt.savefig(save_dir + 'series_' + label.replace(' ','_') + '.png')
        plt.close('all')

        # Get whole period FFT
        f, Pxx = sp.signal.welch(x, fs=fs, nperseg=fs)

        # Make figure for FFT
        plt.figure()
        plt.plot(f, Pxx)
        plt.xlim([0,40])
        plt.title(label + 'FFT')
        
        # Save
        plt.savefig(save_dir + 'FFT_' + label.replace(' ','_') + '.png')
        plt.close('all')