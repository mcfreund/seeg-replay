require(tidyverse)
require(tibble)
require(dplyr)
require(plotrix)
require(ggplot2)
library(patchwork)

# LOAD DATA ####################################################################

df_behavior <- read.csv("/oscar/data/brainstorm-ws/megagroup_data/behavioral_data.csv")
df_behavior$session <- factor(df_behavior$session, 
                              levels = c("Encoding", "SameDayRecall", "NextDayRecall"))
df_behavior$trial_num <- factor(df_behavior$trial_num)
df_behavior$condition <- factor(df_behavior$condition)

participant_ids <- unique(df_behavior$participant_id)
n_participants <- length(participant_ids)


# SANITY CHECKS  ###############################################################

## Videos ----------------------------------------------------------------------
video_by_trial <- df_behavior %>%
  select(participant_id, session, trial_num, condition) %>%
  group_by(participant_id, session) %>%
  pivot_wider(names_from = trial_num, values_from = condition)

video_list <- df_behavior %>%
  select(participant_id, session, trial_num, condition) %>%
  group_by(participant_id, session) %>%
  reframe(movie_list = list(sort(condition)))

# observations: 
# 1) e0013LW has 45 movies; all else have 30
# 2) e0010GP and e0011XQ only do first half of movies on SameDayRecall
# 3) e0010GP and e0011XQ do SameDayRecall and NextDayRecall in the same order across
# sessions and as each other

## Date and Time ---------------------------------------------------------------

date_and_times <- df_behavior %>%
  select(participant_id, session, trial_date_time, trial_num) %>%
  filter(trial_num == 1) %>% 
  select(-trial_num) %>%
  separate(trial_date_time, into = c("Year", "Month", "Day", "Hour", "Min", "Sec"), sep = ",", 
           convert = TRUE, extra = "drop", fill = "right") 
# manually verify year, month are the same for all sessions


# verify that Encoding/SameDay are Same Day
# verify that Encoding = NextDay - 1 
# verify that Encoding Hour = SameDay Hour - 1
# OR
# Encoding Hour = SameDay Hour, Encoding Min < SameDay Min
verify_session_order <- date_and_times %>%
  select(-c(Year, Month, Sec)) %>% # I manually verified these, but TODO add functionality if more participants
  group_by(participant_id) %>%
  pivot_wider(names_from = session, values_from = c(Day, Hour, Min)) %>%
  select(-c(Hour_NextDayRecall, Min_NextDayRecall)) %>%
  mutate(encoding_sameday_day_same = ifelse(Day_Encoding == Day_SameDayRecall, TRUE, FALSE)) %>%
  mutate(encoding_day_before_nextday = ifelse(Day_Encoding == (Day_NextDayRecall - 1), TRUE, FALSE)) %>%
  mutate(encoding_sameday_hr_same = ifelse(Hour_Encoding == Hour_SameDayRecall, TRUE, FALSE)) %>%
  mutate(encoding_hr_less_sameday_hr = ifelse(Hour_Encoding == (Hour_SameDayRecall - 1), TRUE, FALSE)) %>%
  mutate(encoding_min_less_sameday_min = ifelse(Min_Encoding < Min_SameDayRecall, TRUE, FALSE)) %>%
  mutate(order_verified = ifelse(encoding_sameday_day_same & encoding_day_before_nextday &
                                   ((encoding_sameday_hr_same & encoding_min_less_sameday_min) | (!encoding_sameday_hr_same & encoding_hr_less_sameday_hr)), 
                                   TRUE, FALSE))


# CHANGES FROM ENCODING ###############################################################
error_changes <- df_behavior %>%
  select(c(participant_id, session, condition, error_colorpos, error_position)) %>%
  group_by(participant_id, condition) %>%
  pivot_wider(names_from = session, values_from = c(error_colorpos, error_position)) %>%
  mutate(EtoSD_error_position_change = abs(abs(error_position_Encoding) - abs(error_position_SameDayRecall))) %>%
  mutate(EtoSD_error_colorpos_change = abs(abs(error_colorpos_Encoding) - abs(error_colorpos_SameDayRecall))) %>%
  mutate(EtoND_error_position_change = abs(abs(error_position_Encoding) - abs(error_position_NextDayRecall))) %>%
  mutate(EtND_error_colorpos_change = abs(abs(error_colorpos_Encoding) - abs(error_colorpos_NextDayRecall))) %>%
  mutate(SDtoND_error_position_change = abs(abs(error_position_SameDayRecall) - abs(error_position_NextDayRecall))) %>%
  mutate(SDtoND_error_colorpos_change = abs(abs(error_colorpos_SameDayRecall) - abs(error_colorpos_NextDayRecall)))

ggplot(error_changes) +
  geom_point(aes(x = abs(error_position_SameDayRecall), abs(error_position_NextDayRecall))) +
  facet_wrap(~participant_id)

ggplot(error_changes) +
  geom_point(aes(x = abs(error_colorpos_SameDayRecall), abs(error_colorpos_NextDayRecall))) +
  facet_wrap(~participant_id)

ggplot(error_changes) +
  geom_point(aes(x = abs(EtoSD_error_position_change), abs(error_position_NextDayRecall))) +
  facet_wrap(~participant_id)
               

ggplot(error_changes) +
  geom_point(aes(x = condition, y = EtoSD_error_position_change)) +
  geom_line(aes(group = participant_id, x = condition, y = EtoSD_error_position_change), color = "red") +
  
  geom_point(aes(x = condition, y = EtoND_error_position_change)) +
  geom_line(aes(group = participant_id, x = condition, y = EtoND_error_position_change), color = "green") +

  facet_wrap(~participant_id) +
  theme_bw() +
  coord_cartesian(ylim = c(0,180)) +
  scale_y_continuous(breaks = seq(0, 180, by = 45)) + 
  labs(title = "Position Error Changes")

ggplot(error_changes) +

  geom_point(aes(x = condition, y = EtoSD_error_colorpos_change)) +
  geom_line(aes(group = participant_id, x = condition, y = EtoSD_error_colorpos_change), color = "blue") +
  
  geom_point(aes(x = condition, y = EtND_error_colorpos_change)) +
  geom_line(aes(group = participant_id, x = condition, y = EtND_error_colorpos_change), color = "orange") +
  
  facet_wrap(~participant_id) +
  theme_bw() +
  coord_cartesian(ylim = c(0,180)) +
  scale_y_continuous(breaks = seq(0, 180, by = 45)) + 
  labs(title = "Color Error Changes")



# MEMORY METRICS ###############################################################

## High/Low Encoding ----------------------------------------------------------
encoding_strength <- df_behavior %>%
  filter(session == "Encoding") %>%
  mutate(color_encoding_strength = ifelse(abs(error_color) > 1.5, "LOW", "HIGH")) %>%
  mutate(position_encoding_strength = ifelse(abs(error_position) > 1.5, "LOW", "HIGH")) %>%
  select(participant_id, condition, color_encoding_strength, position_encoding_strength)

df_behavior <- left_join(df_behavior,encoding_strength)

## Sleep improvement ----------------------------------------------------------
# Did sleep improve performance?

thresh = 0.5 #defines threshold for epsilon changes in memory

sleep_improvement <- df_behavior %>%
  filter(session != "Encoding") %>%
  select(participant_id, session, condition, error_color, error_position) %>%
  pivot_wider(names_from = session, values_from = c(error_color, error_position)) %>%
  mutate(sleep_improved_color = ifelse(abs(error_color_SameDayRecall) > 
                                         abs(error_color_NextDayRecall) + thresh, T, F)) %>%
  mutate(sleep_improved_position = ifelse(abs(error_position_SameDayRecall) > 
                                            abs(error_position_NextDayRecall) + thresh, T, F)) %>%
  mutate(sleep_improved_both = sleep_improved_color & sleep_improved_position) %>%
  select(participant_id, condition, sleep_improved_color, sleep_improved_position,sleep_improved_both)

df_behavior <- left_join(df_behavior,sleep_improvement)
df_behavior$sleep_improved_both  <- factor(df_behavior$sleep_improved_both, levels = c(FALSE, TRUE))


## Sleep maintenance ----------------------------------------------------------
# Did sleep maintain SameDayRecall level performance?

sleep_maintenance <- df_behavior %>%
  filter(session != "Encoding") %>%
  select(participant_id, session, condition, error_color, error_position) %>%
  pivot_wider(names_from = session, values_from = c(error_color, error_position)) %>%
  mutate(sleep_maintained_color = 
           ifelse((abs(error_color_SameDayRecall) > abs(error_color_NextDayRecall) - thresh) 
                  & (abs(error_color_SameDayRecall) < abs(error_color_NextDayRecall) + thresh) , 
                  T, F)) %>%
  mutate(sleep_maintained_position = 
           ifelse((abs(error_position_SameDayRecall) > abs(error_position_NextDayRecall) - thresh) 
                  & (abs(error_position_SameDayRecall) < abs(error_position_NextDayRecall) + thresh) , 
                  T, F))  %>%
  mutate(sleep_maintained_both = sleep_maintained_color & sleep_maintained_position) %>%
  select(participant_id, condition, sleep_maintained_color, sleep_maintained_position, sleep_maintained_both)

df_behavior <- left_join(df_behavior,sleep_maintenance)
df_behavior$sleep_maintained_both  <- factor(df_behavior$sleep_maintained_both, levels = c(FALSE, TRUE))


# ACCURACY OVERVIEW ############################################################

# filter specific subset
rm(accuracy_to_plot)
accuracy_to_plot <- filter(df_behavior, 
                           #color_encoding_strength == "HIGH",
                           #position_encoding_strength == "HIGH",
                           #sleep_improved_color,
                           #sleep_improved_position,
                           # !is.na(sleep_improved_both),
                           # !is.na(sleep_maintained_both),
                           )

#coloring options: sleep_[improved/maintained]_[color/position/both]
coloring <- "sleep_maintained_both" 

## Color -----------------------------------------------------------------------
# box plot overview
ggplot(accuracy_to_plot, aes(x = as.factor(session), y = (error_color))) + 
  geom_boxplot() + 
  geom_point(position = position_jitter(width = 0.2), alpha = 0.5) +
  theme_bw(20) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  facet_wrap(~participant_id) + 
  labs(title = "Color Error", x = "Session")

# grouped by movie
ggplot(accuracy_to_plot, aes(x = as.factor(session), 
                             y = abs(error_color),
                             color = get(coloring)
                             )) + 
  geom_line(aes(group = condition)) +
  theme_bw(20) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  facet_wrap(~participant_id) + 
  labs(title = "Color Error", x = "Session", 
       color = coloring
       )


## Position  -------------------------------------------------------------------
# box plot overview
ggplot(accuracy_to_plot, aes(x = as.factor(session), y = (error_position))) + 
  geom_boxplot() + 
  geom_point(position = position_jitter(width = 0.2), alpha = 0.5) +
  theme_bw(20) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  facet_wrap(~participant_id) + 
  labs(title = "Positon Error", x = "Session")

# grouped by movie
ggplot(accuracy_to_plot, aes(x = as.factor(session), 
                             y = abs(error_position),
                             color = get(coloring)
                             )) + 
  geom_line(aes(group = condition)) +
  theme_bw(20) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  facet_wrap(~participant_id) + 
  labs(title = "Position Error", x = "Session", 
       color = coloring
       )


# Interactions  ############################################################

rm(sleep_conjunctive)
sleep_conjunctive <- df_behavior %>%
  filter(session == "Encoding") %>%
  filter(!is.na(sleep_improved_color), !is.na(sleep_improved_position)) %>%
  select(participant_id, condition, sleep_improved_color, sleep_improved_position) 

ggplot(sleep_conjunctive, aes(x = sleep_improved_color, y = sleep_improved_position)) +
  geom_point(position = position_jitter(width = 0.2, height = 0.2)) +
  theme_bw(20) +
  facet_wrap(~participant_id)


# I've done binary - it did or didn't - but can also look at the parametric part of the memory

# position x color accuracy parametrically



  















