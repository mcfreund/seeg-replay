%% Extract detected ripple events
% Based off vandermeerlab/code-matlab/example_workflows/LFPeventDetection.m

%% load data
clear all;

%% directory things for toolbox
SET_github_root = '/oscar/home/jhewson/brainstorm/seeg-replay/src/swr';
SET_data_root = '/oscar/home/jhewson/brainstorm/seeg-replay/src/swr';

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
data.e0010GP.CMHIP2.ca1_contact.tstart = lgp_evt.tstart;
data.e0010GP.CMHIP2.ca1_contact.tend = lgp_evt.tend;

%%
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
            data.participant.contact.ca1_contact.tstart = lgp_evt.tstart;
            data.participant.contact.ca1_contact.tend = lgp_evt.tend;
        end
    end
end



%%

function extract_swr_events(data,part,contact)
    disp('EXTRACTING')
    disp(part)
    disp(contact)
    %part = data{i};
    %contact = part{j};
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
    data.participant.contact.ca1_contact.tstart = lgp_evt.tstart;
    data.participant.contact.ca1_contact.tend = lgp_evt.tend;
end
