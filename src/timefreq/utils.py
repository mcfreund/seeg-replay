def load_timefreq_h5(subject, session, params, paths):
    file_path = f'{paths.processed_raws}/{subject}/{session}/{subject}_{session}_{params.suffix_preproc}.h5'
    with h5py.File(file_path, 'r') as hdf_file:
        print("Groups in the HDF5 file:", list(hdf_file.keys()))
        epochs = hdf_file['epochs'][:]
        dimnames = hdf_file['dimnames'][:].astype(str)
        freqs = hdf_file['freqs'][:].astype(str).astype(float)
        times = hdf_file['times'][:].astype(str).astype(float)
        trial_num = hdf_file['trial_num'][:].astype(str).astype(int)
        return epochs, dimnames, freqs, times, trial_num



def plot_time_frequency_heatmap(selected_data, freqs, times, title_str, show=True, filename=None, figsize=(10, 6), grid_size=None, subplot_index=None):
    """
    Plots a time-frequency heatmap for a given trial and channel. Optionally, plots a grid of heatmaps.

    Parameters:
    - selected_data: ndarray, the data array with dimensions (time, frequency)
    - freqs: array-like, the frequency values corresponding to the data
    - times: array-like, the time values corresponding to the data
    - grid_size: tuple (rows, cols), the grid size for plotting multiple heatmaps. If None, plots a single heatmap.
    - subplot_index: int, the index of the subplot in the grid. Ignored if grid_size is None.
    """
    import matplotlib.pyplot as plt

    freqs = [float(x) for x in freqs]

    if grid_size is None:
        # Set up the plot for a single heatmap
        plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        # Set up the plot for a grid of heatmaps
        if subplot_index is None:
            raise ValueError("subplot_index must be provided when grid_size is specified")
        plt.subplot(grid_size[0], grid_size[1], subplot_index + 1)
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(selected_data.T, aspect='auto', origin='lower', extent=[times[0], times[-1], freqs[0], freqs[-1]])

    # Set the y-axis to log scale
    ax.set_yscale('log')

    # Set the y-axis ticks to the frequency values and labels
    freq_labs = [1, 4, 8, 12, 16, 30, 70, 150]
    ax.set_yticks(freq_labs)
    ax.set_yticklabels([f'{freq:.1f}' for freq in freq_labs])

    # Add labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title_str)

    # Add colorbar
    plt.colorbar(im, ax=ax, label='ln(power)')

    # Save the plot
    if filename is not None and grid_size is None:
        plt.savefig(filename)

    if show and grid_size is None:
        plt.show()
