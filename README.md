# Brainstorm Challenge 2024 Repository

## Submission
The state of our project is summarized [here](https://docs.google.com/document/d/1uzaDUmrcNabS9Y7WW-nJzn2VQ5fxtFguiGgj0gWNvyg/edit?usp=sharing). The document is basically a progress report, including details on what we did, how, and why. It also includes information on planned analyses, and our general strategy. We will move it in the direction of a master-document for several conference or journal submissions over time.

## Repository Structure
Plotted data and analyses are in:
```
'./figs/behavior'            : Analysis of subjects behavioral data
'./figs/channels'            : Channel locations and coverage for each subject
'./figs/neural'              : Model-free neural data analysis
'./figs/raw'                 : Preprocessing and data-file agglomeration analyses
```

Behavioral data processing files are:
```
'./src/behavior/analysis.py'                  : Loads behavioral data and generates plots in /figs/behavior
'./src/behavior/behavior_overview.R'          : Similar function as analysis.py
'./src/behavior/restructure_behavioral_data.m': Puts behavioral data from _A.mat and _B.mat files in more useful format
```

Pytorch model files for neural data prediction are:
```
'./src/models/loader.py'     : Contains functions for getting data into proper format for model training/testing
'./src/models/models.py'     : Contains pytorch network models
'./src/models/run.ipy'       : Contains code for producing our analyses
'./src/tmp.py'               : Snippets file for undeveloped and in-process code
```

Model-free neural data analysis code is:
```
'./src/neural/plots.py'      : Plotting functions for plots under ./figs/neural
'./src/neural/run.py'        : Script for generating said plots
```

Preprocessing code can be found in:
```
'./src/preproc/functions.py' : These are the main preprocessing functions (see file header for details)
'./src/preproc/params.py'    : Preprocessing parameters (equivalent to a config.json)
'./src/preproc/run.ipy'      : Script which keeps the full pipeline in order
'./src/preproc/utils.py'     : Subsidiary utilities for the main functions in functions.py
```

Miscellaneous shared code:
```
'./src/shared/paths.py'      : Path predefinitions for different computers. Again, basically a config.json.
'./src/shared/utils.py'      : Data manipulation utilities that don't belong in any more specific place
```

Sharp-wave ripple and oscillatory event-detection code:
```
'./src/swrs/initial/extract_ca1_data.py'       : Original file for getting CA1 data into matlab format
'./src/swrs/initial/vl-shared/'                : Copy of Van der Meer lab SWR toolbox
'./src/swrs/initial/LFPeventDetection_SEEG.m/' : Matlab code for using Van der Meer toolbox on extract_ca1_data.py output
'./src/swrs/initial/riple_events.py/'          : Computes ripple event statistics
'./src/swrs/initial/visualize_events.py/'      : File for showing ripples

'./src/swrs/functions_ca1.py'                  : Aggregates LFP data from electrodes for processing
'./src/swrs/functions_swr.py'                  : Functions for computing oscillatory events
'./src/swrs/misc.py'                           : Code for printing SWR times, comparing pipelines, other miscellany.
'./src/swrs/run.py'                            : Script for running oscillatory event detection
```
