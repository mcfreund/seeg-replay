%% setup
% Brainstorm Challenge 2024
% Author: Alana Jaskir, Brown University
% Last update: Jan 25, 2024
% Description: Reformats relevant behavioral data into tabular format

%clear all;

function [] = restructure_behavioral_data()

     SET_data_root = '/oscar/data/brainstorm-ws/seeg_data/Memory Task Data/Epilepsy/Monitoring/'; % replace this with the location of your local data folder
     SET_save_root = '/oscar/data/brainstorm-ws/megagroup_data/';
     addpath(genpath(SET_data_root)); % clone vandermeerlab repo at https://github.com/vandermeerlab/vandermeerlab

     %% Manually define participant IDs - from Chad Williams 
     % AJ note: I removed participants that did not have next day recall

     participants_of_interest = struct();
     participants_of_interest.e0010GP.day1 = 'e0010GP_00';
     participants_of_interest.e0010GP.day2 = 'e0010GP_01';

     participants_of_interest.e0011XQ.day1 = 'e0011XQ_00';
     participants_of_interest.e0011XQ.day2 = 'e0011XQ_01';

     participants_of_interest.e0017MC.day1 = 'e0017MC_00';
     participants_of_interest.e0017MC.day2 = 'e0017MC_01';

     participants_of_interest.e0019VQ.day1 = 'e0019VQ_00';
     participants_of_interest.e0019VQ.day2 = 'e0019VQ_01';

     participants_of_interest.e0020JA.day1 = 'e0020JA_00';
     participants_of_interest.e0020JA.day2 = 'e0020JA_01';

     participants_of_interest.e0024DV.day1 = 'e0024DV_00';
     participants_of_interest.e0024DV.day2 = 'e0024DV_01';

     participants_of_interest.e0013LW.day1 = 'e0013LW_02';
     participants_of_interest.e0013LW.day2 = 'e0013LW_03';

     participants_of_interest.e0015TJ.day1 = 'e0015TJ_01';
     participants_of_interest.e0015TJ.day2 = 'e0015TJ_02';

     %% Get relevant folder names in data directory

     file_info = dir(SET_data_root);
     is_directory = [file_info.isdir] == 1;
     folder_names = {file_info(is_directory).name};
     folder_names = folder_names(3:end)'; % removes . and .. directories

     %% Restructure behavior into tabular format

     sessions = {'Encoding', 'SameDayRecall', 'NextDayRecall'};

     % initialize data table
     participant_id = [];
     session = [];
     trial_num = [];
     trial_date_time = [];
     condition = [];
     error_colorpos = [];
     error_position = [];
     rt_color = [];
     rt_location = [];
     chosen_colorpos = [];
     chosen_position = [];
     true_position = [];
     true_colorpos = [];
     true_RGB = [];
     true_hue = [];
     chosen_RGB = [];
     chosen_hue = [];
     error_hue = [];

     % Define data table
     data_table=table(...
          participant_id, session, trial_num, trial_date_time, condition, error_colorpos, error_position, ...
          rt_color, rt_location, chosen_colorpos, chosen_position, true_position, ...
          true_colorpos, true_RGB, true_hue, chosen_RGB, chosen_hue, error_hue);
     n_columns = width(data_table);

     % loop through participants
     fs = fields(participants_of_interest);
     for i=1:length(fs)
     this_participant = fs(i);

     % get folder names for day 1 and 2
     day_1 = participants_of_interest.(this_participant{1}).day1;
     day_2 = participants_of_interest.(this_participant{1}).day2;

     day_1_dir = folder_names(cellfun(@(x) ~isempty(regexp(x, strcat(day_1, "$"), 'once')), folder_names));
     day_2_dir = folder_names(cellfun(@(x) ~isempty(regexp(x, strcat(day_2, "$"), 'once')), folder_names));

     day_1_mat_root = day_1_dir{1}(1:end-3);
     day_2_mat_root = day_2_dir{1}(1:end-3);

     % loop through sessions
     for s_idx=1:length(sessions)

          % get appropriate day and .mat ext for condition
          if (s_idx == 1)             % Encoding
               dir_root = day_1_dir;
               mat_root = day_1_mat_root;
               mat_ext  = '_A.mat';
          elseif (s_idx == 2)         % SameDayRecall
               dir_root = day_1_dir;
               mat_root = day_1_mat_root;
               mat_ext  = '_B.mat';
          else                        % NextDayRecall
               dir_root = day_2_dir;
               mat_root = day_2_mat_root;
               mat_ext  = '_A.mat';
          end
          
          try
               load_me = strcat(SET_data_root, dir_root, '/', mat_root, mat_ext);
               load_me = load_me{1};
               load(load_me);
          catch
               % patient e0015Tj has _C for day 2...
               % TODO check day 1 and 2 are indeed a day apart
               load_me = strcat(SET_data_root, dir_root, '/', mat_root, '_C.mat');
               load_me = load_me{1};
               load(load_me);
          end

          % loop through trials
          n_trials = TrialRecord.CurrentTrialNumber;
          for trial_counter=1:n_trials
               
               % make temp row to fill
               temp_row=cell(1,n_columns);
               temp_row{1}=this_participant{1};
               temp_row{2}=sessions{s_idx};
               temp_row{3}=trial_counter;
     
               trial_str =strcat(['Trial' num2str(trial_counter)]);
     
               if isfield(eval(trial_str), 'TrialDateTime')
                    temp_row{4}=strjoin(string(eval(trial_str).TrialDateTime),',');
               else
                    temp_row{4}=nan;
               end
     
               if isfield(eval(trial_str), 'Condition')
                    temp_row{5}=eval(trial_str).Condition;
               else
                    temp_row{5}=nan;
               end
     
               
               if isfield(eval(trial_str).UserVars, 'error_color')
                    error_colorpos = eval(trial_str).UserVars.error_color;
                    temp_row{6}= wrapTo180(rad2deg(error_colorpos));
               else
                    temp_row{6}=nan;
               end
     
               if isfield(eval(trial_str).UserVars, 'error_position')
                    error_position = eval(trial_str).UserVars.error_position;
                    temp_row{7} = wrapTo180(rad2deg(error_position));
               else
                    temp_row{7}=nan;
               end
     
               if isfield(eval(trial_str).UserVars, 'rxn_time_color')
                    temp_row{8}=eval(trial_str).UserVars.rxn_time_color;
               else
                    temp_row{8}=nan;
               end
     
               if isfield(eval(trial_str).UserVars, 'rxn_time_position')
                    temp_row{9}=eval(trial_str).UserVars.rxn_time_position;
               else
                    temp_row{9}=nan;
               end

               if isfield(eval(trial_str).UserVars, 'chosen_color')
                    chosen_colorpos = (eval(trial_str).UserVars.chosen_color);
                    temp_row{10} = wrapTo180(rad2deg(chosen_colorpos));
               else
                    temp_row{10}=nan;
               end

               if isfield(eval(trial_str).UserVars, 'chosen_position')
                    chosen_position = (eval(trial_str).UserVars.chosen_position);
                    temp_row{11} = wrapTo180(rad2deg(chosen_position));
               else
                    temp_row{11}=nan;
               end

               if isfield(eval(trial_str).UserVars, 'chosen_position') * isfield(eval(trial_str).UserVars, 'error_position')
                    temp_row{12} = wrapTo180(rad2deg(error_position + chosen_position));  % true_position
               else
                    temp_row{12}=nan;
               end

               if isfield(eval(trial_str).UserVars, 'chosen_color') * isfield(eval(trial_str).UserVars, 'error_color')
                    temp_row{13} = wrapTo180(rad2deg(error_colorpos + chosen_colorpos));  % true_colorpos
               else
                    temp_row{13}=nan;
               end

               % true_RGB, true_hue
               if isfield(eval(trial_str).UserVars, 'true_RGB')
                    true_RGB = eval(trial_str).UserVars.true_RGB;
                    temp_row{14} = strjoin(string(true_RGB),',');
                    true_hue = rgb_to_lab_hue(true_RGB);
                    temp_row{15} = true_hue;  % true_hue
               else
                    temp_row{14:15}=nan;
               end

               % chosen_RGB, chosen_hue
               if isfield(eval(trial_str).UserVars, 'chosen_RGB')
                    chosen_RGB = eval(trial_str).UserVars.chosen_RGB;
                    temp_row{16} = strjoin(string(chosen_RGB),',');
                    chosen_hue = rgb_to_lab_hue(chosen_RGB);
                    temp_row{17} = chosen_hue;  % chosen_hue
               else
                    temp_row{16:17}=nan;
               end

               % error_hue
               if isfield(eval(trial_str).UserVars, 'chosen_RGB') * isfield(eval(trial_str).UserVars, 'true_RGB')
                    temp_row{18} = wrapTo180(true_hue - chosen_hue);
               else
                    temp_row{18}=nan;
               end

           
               % add to data_table
               data_table=[data_table;temp_row];
     
          end

     end
     end

     %% save to csv
     % in current folder
     %writetable(data_table, 'behavioral_data.csv')
     % in data dir
     filename_out = strcat(SET_save_root,'behavioral_data.csv');
     writetable(data_table, filename_out);
     fileattrib(filename_out, '+w', 'a');  % give write permissions

end


function [hue_angle] = rgb_to_lab_hue(rgb_matrix)
    % Convert from sRGB to CIELAB color space
    rgb_matrix = double(rgb_matrix) / 255;
    lab = rgb2lab(rgb_matrix);
    
    % Calculate the hue angle in radians, convert to degree.
    % NB: hue_angle is already wrapped to [-180, 180]
    % atan2 is used for a better quadrant determination
    hue_angle = rad2deg(atan2(lab(:, 3), lab(:, 2)));
end
