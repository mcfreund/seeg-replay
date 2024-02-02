%% Detecting SWR in LFP in SEEG for Brainstorm2024 Challenge
% Based off vandermeerlab/code-matlab/example_workflows/LFPeventDetection.m

%% load data
clear all;

%% directory things for toolbox
SET_github_root = '/oscar/home/ajaskir/brainstorm_2024/seeg-replay/src/swr';
SET_data_root = '/oscar/home/ajaskir/brainstorm_2024/seeg-replay/src/swr';

restoredefaultpath;
addpath(genpath(cat(2,SET_github_root,'/vandermeerlab/code-matlab/shared'))); % clone vandermeerlab repo at https://github.com/vandermeerlab/vandermeerlab
cd(SET_data_root)

%% set things up
data_file = '/oscar/home/ajaskir/brainstorm_2024/seeg-replay/src/swr/ca1_data_matrix.mat';
load(data_file)

% pick one electrode for now
test_data = data.e0010GP.CMHIP2.ca1_contact;
lfp = test_data;
lfp.cfg.history.mfun = {};   % filler
lfp.cfg.history.cfg = {};    % filler

%%
% create tsd object that contains the fields specified in
% /vandermeerlab/code-matlab/shared/datatypes/tsd/CheckTSD.m

lp = LFPpower([],lfp);
lgp_z = zscore_tsd(lp);

%% detect events
cfg = [];
cfg.method = 'raw';
cfg.threshold = 3;
cfg.dcn =  '>'; % return intervals where threshold is exceeded
cfg.merge_thr = 0.05; % merge events closer than this
cfg.minlen = 0.05; % minimum interval length

lgp_evt = TSDtoIV(cfg,lgp_z);

%% to each event, add a field with the max z-scored power (for later selection)
cfg = [];
cfg.method = 'max'; % 'min', 'mean'
cfg.label = 'maxlgp'; % what to call this in iv, i.e. usr.label
cfg.label = 'maxSWRpower_z'; % what to call this in iv, i.e. usr.label

lgp_evt = AddTSDtoIV(cfg,lgp_evt,lgp_z);

%% select only those events of >5 z-scored power
cfg = [];
cfg.operation = '>';
cfg.threshold = 5;

lgp_evt = SelectIV(cfg,lgp_evt,'maxSWRpower_z');

%% plot events in highlighted on top of full lfp
PlotTSDfromIV([],lgp_evt,lfp);

%% ..or the events alone (fixed 200ms window centered at event time)
close all;

cfg = [];
cfg.display = 'iv';
cfg.mode = 'center';
cfg.fgcol = 'k';

PlotTSDfromIV(cfg,lgp_evt,lfp);
%% ..hold on (highlight edges of event on top of previous plot)
cfg = [];
cfg.display = 'iv';
cfg.fgcol = 'r';

PlotTSDfromIV(cfg,lgp_evt,lfp);