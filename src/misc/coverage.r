library(dplyr)
library(data.table)
library(ggplot2)
library(viridis)
library(here)

## settings

theme_set(theme_bw(base_size = 16))
theme_update(strip.background = element_blank())

## paths

dir_base <-  "~/brainstorm_2024/megagroup"  ## change to repo base dir
dir_figs <- file.path(dir_base, "figs", "channels")
dir.create(dir_figs, showWarnings = FALSE)
dir_data <- '/oscar/data/brainstorm-ws/megagroup_data/'

## read chinfo files

subjects <- list.files(dir_data, pattern = "e001")
fnames_chinfo <- 
  list.files(dir_data, recursive = TRUE, full.names = TRUE, pattern = "chinfo.csv")
l <- setNames(lapply(fnames_chinfo,  fread), subjects)
names(l) <- subjects

## bind, and create columns useful for plotting
## in decreasing spatial size / hier: structure0, ... structure2

chinfo <-
  rbindlist(l, id = "subject", fill = TRUE) %>%
  rename(
    hemi = `Level 0: hemisphere`,
    lobe = `Level 1: lobe`,
    region = `Level 2: region`,
    subregion = `Level 3: gyrus/sulcus/cortex/nucleus`,
    label = `Anatomical Label`,
    wm = `White Matter`
  ) %>%
  mutate(
    hemi = gsub("^ | $", "", hemi),
    lobe = gsub("^ | $", "", lobe),
    region = gsub("^ | $", "", region),
    subregion = gsub("^ | $", "", subregion),
    label = gsub("^ | $", "", label),
    Unit = gsub("^ | $", "", Unit),
    unit = gsub("R |L ", "", Unit),
    ## neocortex versus midbrain:
    structure0 = case_when(
      grepl("hippocampus|amygdala", subregion) | (lobe == "thalamus") ~ "midbrain",
      TRUE ~ lobe
    ),
    ## nucleus / large cortical division
    structure1 = case_when(
      lobe == "frontal" & region %in% c("superior", "middle", "inferior", "central") ~ "LFC",
      lobe == "frontal" & region %in% c("orbitofrontal") ~ "OFC",
      subregion %in% c("superior temporal gyrus", "middle temporal gyrus", "inferior temporal gyrus") ~ "lat. temp. gy.",
      unit %in% c("entorhinal cortex", "parahippocampal gyrus") ~ "PHG",
      unit == "amygdala" ~ "AMY",
      unit == "hippocampus" ~ "HC",
      unit == "thalamus" ~ "thal.",
      TRUE ~ unit
    ),
    ## subregion
    structure2 = case_when(
      unit %in% c("inferior frontal gyrus", "orbitofrontal cortex") ~ Location,
      grepl("hippocampus|amygdala|thalamus", subregion) ~ gsub("hippocampal | amygdaloid | thalamic ", "", Location),
      structure1 == "temporal gyri" ~ subregion,
      TRUE ~ unit
    )
  )



## plot ----

## remove channels marked bad, in WM, or lost in re-referencing.

chinfo_good <- 
  chinfo %>%
  filter(
    if_all(starts_with("is_bad"), ~ !.),
    ## was_rereferenced_* cols indicate channels that SURVIVED rereferencing:
    if_all(starts_with("was_rereferenced"), ~ .),
    wm == 0
  )


p_structure0_box <- chinfo_good %>%
  count(subject, structure0) %>%
  ggplot() +
  geom_boxplot(aes(x = structure0, y = n), width = 0.25) +
  coord_flip() +
  labs(x = "structure", y = "n_sites")

p_structure0_box

ggsave(
  file.path(dir_figs, "coverage_grossanat_box.pdf"),
  p_structure0_box, device = "pdf", width = 5, height = 3)


p_structure0 <- chinfo_good %>%
  count(subject, hemi, structure0) %>%
  ggplot() +
  geom_point(aes(x = subject, y = structure0, color = n), size = 8) +
  geom_text(aes(x = subject, y = structure0, label = n), size = 3, color = "white") +
  scale_color_viridis_c(option = "magma", limits = c(1, 50)) +
  facet_grid(vars(hemi), scales = "free_y") +
  theme(panel.border = element_blank(), axis.text.x = element_text(angle = 90, vjust = 0.5)) +
  labs(y = NULL, color = "n_sites", title = "coverage per subject: gross anat.")

p_structure0

ggsave(
  file.path(dir_figs, "coverage_grossanat_dot.pdf"),
  p_structure0, device = "pdf", width = 5.5, height = 8.5)


p_structure1 <- chinfo_good %>%
  filter(structure0 %in% c("temporal", "frontal", "midbrain")) %>%
  count(subject, hemi, structure0, structure1) %>%
  ggplot() +
  geom_point(aes(x = subject, y = structure1, color = n), size = 8) +
  geom_text(aes(x = subject, y = structure1, label = n), size = 3, color = "white") +
  scale_color_viridis_c(option = "magma", limits = c(1, 30)) +
  facet_grid(vars(hemi, structure0), scales = "free_y") +
  theme(panel.border = element_blank(), axis.text.x = element_text(angle = 90, vjust = 0.5)) +
  labs(
    y = NULL, color = "n_sites", 
    title = "coverage per subject: region",
    caption = "NB: only orb+lat front, temp ctx, and midbrain,\ngrey matter")

p_structure1

ggsave(
  file.path(dir_figs, "coverage_region_dot.pdf"),
  p_structure1, device = "pdf", width = 6, height = 8)

p_structure2 <- chinfo_good %>%
  filter(
    structure0 %in% c("temporal", "frontal", "midbrain"),
    structure1 %in% c("LFC", "OFC", "AMY", "HC", "lat. temp. gy.", "thal.")) %>%
  count(subject, hemi, structure0, structure1, structure2) %>%
  ggplot() +
  geom_point(aes(x = subject, y = structure2, color = n), size = 8) +
  geom_text(aes(x = subject, y = structure2, label = n), size = 3, color = "white") +
  scale_color_viridis_c(option = "magma", limits = c(1, 15)) +
  facet_grid(vars(hemi, structure0, structure1), scales = "free_y") +
  theme(panel.border = element_blank(), axis.text.x = element_text(angle = 90, vjust = 0.5)) +
  labs(
    y = NULL,
    color = "n_sites",
    title = "coverage per subject: subregion",
    caption = "NB: only orb+lat front, lat temp ctx, and midbrain,\ngrey matter")

p_structure2

ggsave(
  file.path(dir_figs, "coverage_subregion_dot.pdf"),
  p_structure2, device = "pdf", width = 7.5, height = 20)



## channels lost? ----

nrow(chinfo)  ## total channels
nrow(chinfo_good) ## those retained
sum(chinfo$wm)
## channels marked bad:
chinfo %>% filter(if_any(starts_with("is_bad"), ~ .))  ## bad chs
## num channels lost in re-referencing:
chinfo %>%
  summarize(across(starts_with("was_reref"), \(x) sum(1 - x, na.rm = TRUE)))
## channels lost in re-referencing that would not have otherwise been excluded:
chinfo %>%
  filter(
    if_all(starts_with("is_bad"), ~ !.),
    wm == 0,
    if_any(starts_with("was_rereferenced"), ~ !.)
  ) %>%
  View

chinfo %>%
  filter(
    if_all(starts_with("is_bad"), ~ !.),
    wm == 0,
    if_any(starts_with("was_rereferenced"), ~ !.),
    Location == "CA1"
  )
