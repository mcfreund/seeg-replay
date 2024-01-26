# "megagroup" repo for Brainstorm Challenge 2024

https://brainstorm-program.github.io/brainstorm_challenge_2024/

## repo structure

`./src`
- main code for preproc/analyses
- scripts read from local data directory: `/oscar/data/brainstorm-ws/megagroup_data`
- write to `./derivatives`

`./misc`
- notebooks/rmarkdowns, one-off scripts for exploring data, etc.

`./derivatives`
- by default, all contents are ignored. if there is something useful to share with the group here that is less than 50 MB, then override in `.gitignore`.

## local data directory
```
/oscar/data/brainstorm-ws/megagroup_data
```
- subject subdirectories contain:
  - raw timeseries (`_raw`) in two formats -- FIFF and brainvision
  - anatomical info (`_chinfo.csv`)
  - trial events / triggers (`_events.csv`)
  - trial metadata (`events_metadata.csv`)  -- not yet added
