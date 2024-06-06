#!/bin/bash -l

#PBS -N sunerf-cme
#PBS -A P22100000
#PBS -q preempt
#PBS -l select=1:ncpus=8:ngpus=2:mem=24gb
#PBS -l walltime=12:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/sunerf-cme

# Prep 2view
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_340W_bang_0000_pB/*.fits" --output_path /glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_2view
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_340W_bang_0000_tB/*.fits" --output_path /glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_2view
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_280W_bang_0000_pB/*.fits" --output_path /glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_2view
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_280W_bang_0000_tB/*.fits" --output_path /glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_2view

# Prep allview
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_allview"  --check_matching

# Prep 2view_background
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_340W_bang_0000_pB/*.fits" --output_path /glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_2view_background
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_340W_bang_0000_tB/*.fits" --output_path /glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_2view_background
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_280W_bang_0000_pB/*.fits" --output_path /glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_2view_background
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_280W_bang_0000_tB/*.fits" --output_path /glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_2view_background

# Prep 5view
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_340W_bang_0000_*B/*.fits" --output_path /glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_5view --check_matching
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_280W_bang_0000_*B/*.fits" --output_path /glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_5view --check_matching
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_220W_bang_0000_*B/*.fits" --output_path /glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_5view --check_matching
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_100W_bang_0000_*B/*.fits" --output_path /glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_5view --check_matching
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_040W_bang_0000_*B/*.fits" --output_path /glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_5view --check_matching

python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/*_bang_0000_*/stepnum_005.fits" --output_path /glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_2view_background --check_matching

#####################################
#                                   #
#   Download Data Preset Commands   #
#                                   #
#####################################

# Download all viewpoints first into data_fits subdirectory, then download all 005.step files as well.
gsutil -m cp -R gs://fdl23_europe_helio_onground/ground-data/data_fits /mnt/ground-data/

# Download 2 Viewpoints only into data_fits directory, emulating L5 and Earth (60° Diff) /glade/work/rjarolim/data/sunerf-cme/hao
gsutil -m cp -R  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_340W_bang_0000_tB /glade/work/rjarolim/data/sunerf-cme/hao/
gsutil -m cp -R  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_340W_bang_0000_pB /glade/work/rjarolim/data/sunerf-cme/hao/

gsutil -m cp -R  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_280W_bang_0000_tB /glade/work/rjarolim/data/sunerf-cme/hao/
gsutil -m cp -R  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_280W_bang_0000_pB /glade/work/rjarolim/data/sunerf-cme/hao/


# Download Background data into /glade/work/rjarolim/data/sunerf-cme/hao
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_020W_bang_0000_tB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_020W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_020W_bang_0000_pB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_020W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_040W_bang_0000_tB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_040W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_040W_bang_0000_pB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_040W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_060W_bang_0000_tB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_060W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_060W_bang_0000_pB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_060W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_080W_bang_0000_tB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_080W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_080W_bang_0000_pB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_080W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_100W_bang_0000_tB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_100W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_100W_bang_0000_pB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_100W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_120W_bang_0000_tB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_120W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_120W_bang_0000_pB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_120W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_140W_bang_0000_tB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_140W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_140W_bang_0000_pB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_140W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_160W_bang_0000_tB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_160W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_160W_bang_0000_pB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_160W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_180W_bang_0000_tB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_180W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_180W_bang_0000_pB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_180W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_200W_bang_0000_tB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_200W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_200W_bang_0000_pB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_200W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_220W_bang_0000_tB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_220W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_220W_bang_0000_pB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_220W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_240W_bang_0000_tB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_240W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_240W_bang_0000_pB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_240W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_260W_bang_0000_tB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_260W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_260W_bang_0000_pB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_260W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_280W_bang_0000_tB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_280W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_280W_bang_0000_pB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_280W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_300W_bang_0000_tB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_300W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_300W_bang_0000_pB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_300W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_320W_bang_0000_tB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_320W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_320W_bang_0000_pB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_320W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_340W_bang_0000_tB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_340W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_340W_bang_0000_pB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_340W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_360W_bang_0000_tB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_360W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_360W_bang_0000_pB/stepnum_005.fits /glade/work/rjarolim/data/sunerf-cme/hao/dcmer_360W_bang_0000_pB/stepnum_005.fits

# Download for all of the PSI Data
gsutil -m cp -R gs://fdl23_europe_helio_onground/ground-data/PSI /mnt/ground-data/PSI/

# Download observational data
gsutil -m cp -R gs://fdl_space_weather_data/events/fdl_stereo_2014_02_prep.zip /mnt/ground-data/

################
#              #
#  Prep Data   #
#              #
################

# Prep all data available - only ever download the data that we would like to use
# Prep_HAO_2view
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/**/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_2view" --check_matching

# Prep_HAO_2view_background
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/**/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_2view_background" --check_matching

# prep_HAO_allview
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/**/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_allview" --check_matching

# PSI
# convert data
python -m sunerf.prep.prep_psi_cor --psi_path "/mnt/ground-data/PSI/pb_raw/*.fits"  --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_PSI/pb_raw"
python -m sunerf.prep.prep_psi_cor --psi_path "/mnt/ground-data/PSI/b_raw/*.fits"  --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_PSI/b_raw"


######################
#                    #
#   Running ICARUS   #
#                    #
######################

# Prep_HAO_1view
python -m sunerf.sunerf --wandb_name "hao_pinn_1view" --data_path_pB "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_1view/*pB*.fits" --data_path_tB "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_1view/*tB*.fits" --path_to_save "/mnt/training/HAO_pinn_1view" --train "config/train.yaml" --hyperparameters "config/hyperparams_hao.yaml"

# Prep_HAO_2view
python -m sunerf.sunerf --wandb_name "hao_pinn_2view" --data_path_pB "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_2view/*pB*.fits" --data_path_tB "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_2view/*tB*.fits" --path_to_save "/mnt/training/HAO_pinn_2view" --train "config/train.yaml" --hyperparameters "config/hyperparams_hao.yaml"
# prep_HAO_2view_backgrounds
python -m sunerf.sunerf --wandb_name "hao_pinn_2view_background" --data_path_pB "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_2view_background/*pB*.fits" --data_path_tB "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_2view_background/*tB*.fits" --path_to_save "/mnt/training/HAO_pinn_2view_background" --train "config/train.yaml" --hyperparameters "config/hyperparams_hao.yaml"
# prep_HAO_allview
python -m sunerf.sunerf --wandb_name "hao_pinn_all" --data_path_pB "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_allview/*pB*.fits" --data_path_tB "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_allview/*tB*.fits" --path_to_save "/mnt/training/HAO_pinn_allview" --train "config/train.yaml" --hyperparameters "config/hyperparams_hao.yaml"

# Prep_HAO_5view
python -m sunerf.sunerf --wandb_name "hao_pinn_5view" --data_path_pB "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_5view/*pB*.fits" --data_path_tB "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_5view/*tB*.fits" --path_to_save "/mnt/training/HAO_pinn_5view" --train "config/train.yaml" --hyperparameters "config/hyperparams_hao.yaml"

# full training PSI
python -m sunerf.sunerf --wandb_name "psi" --data_path_pB "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_PSI/pb_raw/*.fits" --data_path_tB "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_PSI/b_raw/*.fits" --path_to_save "/mnt/training/PSI_v2" --train "config/train.yaml" --hyperparameters "config/hyperparams_psi.yaml"

# prep OBS
python -m sunerf.prep.prep_obs --resolution 512

# train OBS
python -m sunerf.sunerf --wandb_name "obs_v2" --data_path_pB "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_OBS/*_1P*.fts" --data_path_tB "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_OBS/*_1B*.fts" --path_to_save "/mnt/training/OBS_v2" --train "config/train.yaml" --hyperparameters "config/hyperparams_obs.yaml"
