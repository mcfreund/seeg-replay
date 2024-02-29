function data = get_swrs_from_lfps(thresh, save_flag, plt)
    % Function:
    %   Reads a data file created by extract_ca1_data.py, runs vandermeer lab code
    %   to filter time-series data and detect sharp-wave ripples, does some QC.
    %
    % Inputs:
    %   thresh   - reject events with proximal contact LFP z-score > thresh
    %   save_flg - save the data
    %   plt      - save plots of the events (non-functional currently)
    % 
    % Outputs:
    %   data - ca1_data_matrix.mat, with SWR events appended as fields
    %
    % Writes (optionally): 
    %   event_ca1_data.mat - a copy of the data that is output
    %   [subj]_[epoch]_[contact]_[events].* - plots of swrs


    % Extract detected ripple events
    % Based off vandermeerlab/code-matlab/example_workflows/LFPeventDetection.m

    % directory things for toolbox
    path_ca1_data = './';
    path_vml_code = './vl-shared';
    
    % Add Van der Meer lab code. Will warn re /vl-shared/datatypes/plot.m
    addpath(genpath(path_vml_code));
    
    % Get subject data and contact info
    data_file = [path_ca1_data,'ca1_data_matrix.mat'];
    load(data_file, 'data')
    load(data_file, 'usable_contacts')
    
    % List of subject names
    subjs = fieldnames(usable_contacts);
    
    % Extract swr events from all CA1 channels
    for i = 1:length(subjs)
    
        % This subject and their contacts. FIX: The contacts list has datatype inconsistency.
        subj = subjs{i};
        subj_contacts = usable_contacts.(subj);
    
        % Check if there are usable contacts
        if not(isempty(subj_contacts))
    
            % Different naming conventions in different places...
            contact = remove_hypthen(subj_contacts);
    
            % Returns contact SWR event starts, ends, power above and below
            swr = get_swrs_from_contact(subj, contact, data, thresh, plt);
    
            % Append to data structure
            data.(subj).(contact).ca1_contact = swr;
        end
    end
    
    % Save data
    if save_flag
        disp('Saving data...')
        save('event_ca1_data.mat','data')
        disp('Data saved.')
    end
end


function swr = get_swrs_from_contact(subj, contact, data, thresh, plt)

    % 
    disp(' ')
    disp(['Getting SWRs for subject ' subj ' from contact ' contact])

    % pick one electrode for now
    lfp_raw       = add_filler_flds(data.(subj).(contact).ca1_contact  ); 
    lfp_raw_above = add_filler_flds(data.(subj).(contact).contact_above);
    lfp_raw_below = add_filler_flds(data.(subj).(contact).contact_below);
    
    % Check if either contact is missing
    no_above = strcmp(lfp_raw_above.data,'EMPTY');
    no_below = strcmp(lfp_raw_below.data,'EMPTY');

    % Vanilla power and z-scoring (from LFPDetection)
    % create tsd object that contains the fields specified in
    % /vandermeerlab/code-matlab/shared/datatypes/tsd/CheckTSD.m

    % Get power from each contact if possible
    [~, lfp_power_z      ] = try_get_power(lfp_raw      , ''     );
    [~, lfp_power_z_above] = try_get_power(lfp_raw_above, 'above');
    [~, lfp_power_z_below] = try_get_power(lfp_raw_below, 'below');

    % Detect events
    swr_evt = event_detection(lfp_power_z);

    % Remove events with elevated power above and below thresh
    % An alternative is running event detection on above and below
    rm = zeros(1, length(swr_evt.tstart));
    if ~no_above; rm = rm + iv_power_check(swr_evt, lfp_power_z_above, thresh); end
    if ~no_below; rm = rm + iv_power_check(swr_evt, lfp_power_z_below, thresh); end

    % Notify
    disp(['Events identified            : ' num2str(length(swr_evt.tstart))])
    disp(['Events identified as spurious: ' num2str(sum(rm > 0.1))])
    
    % TODO: Create plots of all events and adjacent channels here.

    % Save to output
    swr.tstart = swr_evt.tstart(rm <= 0.1);
    swr.tend   = swr_evt.tend(rm <= 0.1);
    swr.pz_above = lfp_power_z_above;
    swr.pz_below = lfp_power_z_below;
end

function wrap_multi_raster()

    % Create fake spike train to get MultiRaster to work
    n_data_fillers = length(lfp_raw.data);
    spike_filler.t = {(1:100:n_data_fillers)'};
    spike_filler.type = 'ts';
    spike_filler.label = {"filler"};

    cfg_mr = []; 
    cfg_mr.lfpMax = Inf; 
    cfg_mr.lfpHeight = 10; 
    cfg_mr.lfpSpacing = 7; 
    cfg_mr.lfpColor = 'k';
    
    cfg_mr.lfp(1) = lfp_raw; 
    cfg_mr.lfp(2) = lfp_power_z; 
    %cfg_mr.lfp(3) = unfiltered_raw; 
    
    cfg_mr.lfp(3) = lfp_power_z_above;
    cfg_mr.lfp(4) = lfp_raw_above;
    %cfg_mr.lfp(5) = lfp_power_z_below; 
    
    cfg_mr.evt = swr_evt;
    MultiRaster(cfg_mr, spike_filler);
    xlim([0,1])


end


function remove = iv_power_check(swr_evt, lfp_power_z, thresh)
    
    % Number of events to check
    n_events = length(swr_evt.tstart);

    % Flags for removal
    remove = zeros(1, n_events);

    % Check each event
    for i = 1:n_events

        % Get location in lfp data
        inds = find(lfp_power_z.tvec_filtered > swr_evt.tstart(i) ...
            & lfp_power_z.tvec_filtered < swr_evt.tend(i));

        % See if reference electrode is elevated
        if mean(lfp_power_z.data(inds)) > thresh
            remove(i) = 1;
        end
    end
end


function swr_evt = event_detection(lfp_power_z)
    % Configuration for candidate interval extraction
    cfg.method = 'raw';
    cfg.threshold = 3; %3
    %cfg.dcn =  '>'; % return intervals where threshold is exceeded
    cfg.merge_thr = 0.05; % merge events closer than this
    cfg.minlen    = 0.025; % minimum interval length
    cfg.verbose   = 0;

    % Get candidate intervals
    swr_evt = TSDtoIV(cfg, lfp_power_z);
    
    % Add max z-scored power to each interval event (for selection at next step)
    cfg.method  = 'max'; % 'min', 'mean'
    cfg.label   = 'maxSWRpower_z'; % what to call this in iv, i.e. usr.label
    cfg.verbose = 0;
    swr_evt     = AddTSDtoIV(cfg,swr_evt,lfp_power_z);
    
    % Selection only those events of >5 z-scored power
    cfg.operation = '>';
    cfg.threshold = 5;
    cfg.verbose   = 0;
    swr_evt = SelectIV(cfg, swr_evt,'maxSWRpower_z');
end

function struct = add_filler_flds(struct)
    % This is silly.
    struct.cfg.history.mfun = {};
    struct.cfg.history.cfg  = {};
end

function [lfp_power, lfp_power_z] = try_get_power(raw, str)
    err = '';
    lfp_power   = [];
    lfp_power_z = [];
    cfg.verbose = 0;
    try
        lfp_power   = LFPpower(cfg,raw); 
        lfp_power_z = zscore_tsd(lfp_power);
    catch
        err = ['No contact ' str];
    end
    disp(err)
end

function idstr = remove_hypthen(idstr)
    % Removes hypthen from contact id string if it's there.
    if idstr(2) == '-'
        idstr = [idstr(1),idstr(3:7)];
    end
end



%% Deprecated

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
