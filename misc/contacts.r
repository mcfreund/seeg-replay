library(dplyr)
library(data.table)
library(ggplot2)
library(readxl)

dir_chinfo <- '/oscar/data/brainstorm-ws/seeg_data/Memory Task Data/Imaging/Epilepsy'
subjects <- list.files(dir_chinfo)
l <- lapply(list.files(dir_chinfo, recursive = TRUE, full.names = TRUE), read_excel)
names(l) <- subjects
chinfo <- bind_rows(l, .id = "subject")
fwrite(chinfo, "chinfo.csv")
chinfo <- chinfo[!is.na(chinfo$`Anatomical Label`), ]

chinfo <- chinfo %>%
  rename(
    hemi = `Level 0: hemisphere`,
    lobe = `Level 1: lobe`,
    region = `Level 2: region`,
    subregion = `Level 3: gyrus/sulcus/cortex/nucleus`,
    label = `Anatomical Label`,
    wm = `White Matter`
  ) %>% 
  mutate(
    hemi_lobe = paste0(hemi, "_", lobe),
    hemi_lobe_region = paste0(hemi, "_", lobe, "_", region),
    hemi_lobe_subregion = paste0(hemi, "_", lobe, "_", region, "_", subregion),
    lobe_region = paste0(lobe, "_", region),
    lobe_subregion = paste0(lobe, "_", region, "_", subregion)
    )


chinfo %>%  
  count(subject, hemi_lobe) %>%
  ggplot() +
  geom_boxplot(aes(x = hemi_lobe, y = n)) +
  coord_flip()
ggsave("boxplot_hemi_lobe.pdf")


chinfo %>%  
  count(subject, hemi, lobe_region) %>%
  ggplot() +
  geom_boxplot(aes(x = lobe_region, y = n)) +
  coord_flip() +
  facet_grid(cols = vars(hemi))
ggsave("boxplot_hemi_lobe_region.pdf")

chinfo %>%
  filter(lobe == "temporal") %>%
  count(subject, hemi, subregion) %>%
  ggplot() +
  geom_boxplot(aes(x = subregion, y = n)) +
  coord_flip() +
  facet_grid(vars(hemi))
ggsave("temporal_subregion_box.pdf")


chinfo %>%
  filter(lobe == "temporal") %>%
  count(subject, hemi, subregion) %>%
  ggplot() +
  geom_poi(aes(x = subregion, y = n)) +
  coord_flip() +
  facet_grid(vars(hemi))



chinfo %>%
  filter(lobe == "temporal") %>%
  count(subject, hemi, lobe, subregion) %>%
  ggplot() +
  geom_point(aes(x = subject, y = subregion, color = n)) +
  facet_grid(vars(lobe, hemi))
chinfo %>%
  filter(lobe %in% c("frontal", "temporal")) %>%
  count(subject, hemi, lobe, subregion) %>%
  ggplot() +
  geom_point(aes(x = subject, y = subregion, color = n)) +
  #facet_grid(vars(lobe, hemi))
  facet_wrap(vars(lobe, hemi), ncol = 1, scales = "free_y")
ggsave("frontal_subregion_dot.pdf")

