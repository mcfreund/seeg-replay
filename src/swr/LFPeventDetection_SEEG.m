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

%% pick one electrode for now
main_contact = data.e0010GP.CMHIP2.ca1_contact; 
above_contact = data.e0010GP.CMHIP2.contact_above;
below_contact = data.e0010GP.CMHIP2.contact_below;

% make filler spike train to use MultiRaster -----------------------------
n_data_fillers = length(main_contact.data);
S_filler.t = {(1:100:n_data_fillers)'};
S_filler.type = 'ts';
S_filler.label = {"filler"};

% "data" field specifies band pass filtered 80-100 hz, ------------------
% so pull out other data for plotting  
unfiltered_raw = main_contact;                   
unfiltered_raw.data = unfiltered_raw.data_unfiltered;  
unfiltered_raw.cfg.history.mfun = {};
unfiltered_raw.cfg.history.cfg = {};

% fill in configuration information -------------------------------------
lfp_raw = main_contact;
lfp_raw.cfg.history.mfun = {};   % filler
lfp_raw.cfg.history.cfg = {};    % filler

% above contact
lfp_raw_above = above_contact;
lfp_raw_above.cfg.history.mfun = {};   
lfp_raw_above.cfg.history.cfg = {};    

% below contact
lfp_raw_below = below_contact;
lfp_raw_below.cfg.history.mfun = {};   
lfp_raw_below.cfg.history.cfg = {};    

%% Vanilla power and z-scoring (from LFPDetection)
% create tsd object that contains the fields specified in
% /vandermeerlab/code-matlab/shared/datatypes/tsd/CheckTSD.m

cfg = []; cfg.output = 'power';
lfp_power = LFPpower([],lfp_raw); % spits out time x power 
lfp_power_z = zscore_tsd(lfp_power);

try
    lfp_power_above = LFPpower([],lfp_raw_above); 
    lfp_power_z_above = zscore_tsd(lfp_power_above);
catch
    err = "No above contact"
end

try
    lfp_power_below = LFPpower([],lfp_raw_below); 
    lfp_power_z_below = zscore_tsd(lfp_power_below);
catch
    err = "No below contact"
end


%% detect events
cfg = [];
cfg.method = 'raw';
cfg.threshold = 3; %3
%cfg.dcn =  '>'; % return intervals where threshold is exceeded
cfg.merge_thr = 0.05; % merge events closer than this
cfg.minlen = 0.025; % minimum interval length

SWR_evt = TSDtoIV(cfg,lfp_power_z);

% to each event, add a field with the max z-scored power (for later selection)
cfg = [];
cfg.method = 'max'; % 'min', 'mean'
cfg.label = 'maxSWRpower_z'; % what to call this in iv, i.e. usr.label
SWR_evt = AddTSDtoIV(cfg,SWR_evt,lfp_power_z);

%% select only those events of >5 z-scored power
cfg = [];
cfg.operation = '>';
cfg.threshold = 5;
SWR_evt = SelectIV(cfg,SWR_evt,'maxSWRpower_z');

%% plot events in highlighted on top of full lfp
%PlotTSDfromIV([],lgp_evt,test_data);

%% plot the events alone (fixed 200ms window centered at event time)
close all;

% cfg = [];
% cfg.display = 'iv';
% cfg.mode = 'center';
% cfg_def.width = 1.0;
% cfg.fgcol = 'r';
% 
% PlotTSDfromIV(cfg,SWR_evt,lfp_raw);

%% Use MultiRaster

cfg_mr = []; 
cfg_mr.lfpMax = Inf; 
cfg_mr.lfpHeight = 10; 
cfg_mr.lfpSpacing = 7; 
cfg_mr.lfpColor = 'k';

cfg_mr.lfp(1) = lfp_raw; 
cfg_mr.lfp(2) = lfp_power_z; 
cfg_mr.lfp(3) = unfiltered_raw; 

cfg_mr.lfp(4) = lfp_power_z_above; 
%cfg_mr.lfp(5) = lfp_power_z_below; 

cfg_mr.evt = SWR_evt;
MultiRaster(cfg_mr, S_filler);
xlim([0,1])

