%% Extract detected ripple events
% Based off vandermeerlab/code-matlab/example_workflows/LFPeventDetection.m

%% load data
clear all;

%% directory things for toolbox
SET_github_root = '/oscar/home/jhewson/brainstorm/seeg-replay/src/swr';
SET_data_root = '/oscar/home/jhewson/brainstorm/seeg-replay/src/swr';
SET_save_root = '/oscar/data/brainstorm-ws/megagroup_data';

restoredefaultpath;
addpath(genpath(cat(2,SET_github_root,'/vandermeerlab/code-matlab/shared'))); % clone vandermeerlab repo at https://github.com/vandermeerlab/vandermeerlab
cd(SET_data_root)

%% set things up
data_file = '/oscar/home/jhewson/brainstorm/seeg-replay/src/swr/ca1_data_matrix.mat';
load(data_file)

%%
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
%data.e0010GP.CMHIP2.ca1_contact.tstart = lgp_evt.tstart;
%data.e0010GP.CMHIP2.ca1_contact.tend = lgp_evt.tend;

%% extract swr events from all CA1 channels
part_names = fieldnames(data);
participant_names = fieldnames(usable_contacts);
for i = 1:length(participant_names)
    part = participant_names{i};
    part_contacts = usable_contacts.(part);
    if size(usable_contacts.(part)) > 0
        for j = 1:size(usable_contacts.(part)(1))
            if part_contacts(j,2) == '-'
                contact = [part_contacts(j,1),part_contacts(j,3:7)];
            else
                contact = part_contacts(j,:);
            end
            %process(part,contact,data)
            AlanaProcess(part,contact,data)
        end
    end
end

%% Save data

restoredefaultpath;
cd(SET_data_root)
%addpath(genpath(SET_save_root)); % clone vandermeerlab repo at https://github.com/vandermeerlab/vandermeerlab
%cd(SET_save_root)
save('event_ca1_data.mat','data')

%% func

function process(part, contact, data)
    % extract_swr_events(data,part,contact)
    lfp = data.(part).(contact).ca1_contact;
    lfp.cfg.history.mfun = {};   % filler
    lfp.cfg.history.cfg = {};    % filler
    lp = LFPpower([],lfp);
    lgp_z = zscore_tsd(lp);
    cfg = [];
    cfg.method = 'raw';
    cfg.threshold = 3;
    cfg.dcn =  '>'; % return intervals where threshold is exceeded
    cfg.merge_thr = 0.05; % merge events closer than this
    cfg.minlen = 0.05; % minimum interval length
    lgp_evt = TSDtoIV(cfg,lgp_z);
    disp(lgp_evt)
    data.(part).(contact).ca1_contact.tstart = lgp_evt.tstart;
    data.(part).(contact).ca1_contact.tend = lgp_evt.tend;
end

%% new approach adapted by Alana

function AlanaProcess(part, contact, data)
    % pick one electrode for now
    main_contact = data.(part).(contact).ca1_contact; 
    above_contact = data.(part).(contact).contact_above;
    below_contact = data.(part).(contact).contact_below;
    % "data" field specifies band pass filtered 80-100 hz, ------------------
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
    % Vanilla power and z-scoring (from LFPDetection)
    % create tsd object that contains the fields specified in
    % /vandermeerlab/code-matlab/shared/datatypes/tsd/CheckTSD.m
    cfg = []; cfg.output = 'power';
    lfp_power = LFPpower([],lfp_raw); % spits out time x power 
    lfp_power_z = zscore_tsd(lfp_power);
    err = "";
    lfp_power_z_above = [];
    lfp_power_z_below = [];
    try
        lfp_power_above = LFPpower([],lfp_raw_above); 
        lfp_power_z_above = zscore_tsd(lfp_power_above);
    catch
        err = "No above contact";
    end
    try
        lfp_power_below = LFPpower([],lfp_raw_below); 
        lfp_power_z_below = zscore_tsd(lfp_power_below);
    catch
        err = "No below contact";
    end
    disp(err)
    % detect events
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
    % select only those events of >5 z-scored power
    cfg = [];
    cfg.operation = '>';
    cfg.threshold = 5;
    SWR_evt = SelectIV(cfg,SWR_evt,'maxSWRpower_z');
        
    data.(part).(contact).ca1_contact.tstart = SWR_evt.tstart;
    data.(part).(contact).ca1_contact.tend = SWR_evt.tend;
    data.(part).(contact).ca1_contact.pz_above = lfp_power_z_above;
    data.(part).(contact).ca1_contact.pz_below = lfp_power_z_below;
end
    