# Training from scratch
# init workspace

# PSI
# convert data
python -m sunerf.prep.prep_psi_cor --psi_path "/mnt/ground-data/PSI/pb_raw/*.fits"  --output_path "/mnt/ground-data/prep_PSI/pb_raw"
python -m sunerf.prep.prep_psi_cor --psi_path "/mnt/ground-data/PSI/b_raw/*.fits"  --output_path "/mnt/ground-data/prep_PSI/b_raw"
# full training PSI
python -m sunerf.sunerf --wandb_name "psi" --data_path_pB "/mnt/ground-data/prep_PSI/pb_raw/*.fits" --data_path_tB "/mnt/ground-data/prep_PSI/b_raw/*.fits" --path_to_save "/mnt/ground-data/training/PSI_v1" --train "config/train.yaml" --hyperparameters "config/hyperparams_icarus.yaml"

# HAO
# convert data
sudo chmod -R 777 /mnt/ground-data
rm -r /mnt/ground-data/prep_HAO
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/mnt/ground-data/data_fits/dcmer_340W_bang_0000_pB/*.fits" --output_path /mnt/ground-data/prep_HAO_2view_background
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/mnt/ground-data/data_fits/dcmer_340W_bang_0000_tB/*.fits" --output_path /mnt/ground-data/prep_HAO_2view_background
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/mnt/ground-data/data_fits/dcmer_280W_bang_0000_pB/*.fits" --output_path /mnt/ground-data/prep_HAO_2view_background
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/mnt/ground-data/data_fits/dcmer_280W_bang_0000_tB/*.fits" --output_path /mnt/ground-data/prep_HAO_2view_background


# python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/mnt/ground-data/**/*.fits" --output_path /mnt/ground-data/prep_HAO_2view_background
# full training

python -m sunerf.sunerf --wandb_name "hao_pinn_2viewpoints" --data_path_pB "/mnt/ground-data/prep_HAO/*pB*.fits" --data_path_tB "/mnt/ground-data/prep_HAO/*tB*.fits" --path_to_save "/mnt/ground-data/training/HAO_pinn_2viewpoint_v3" --train "config/train.yaml" --hyperparameters "config/hyperparams_hao.yaml"
python -m sunerf.sunerf --wandb_name "hao_pinn_allviewpoints" --data_path_pB "/mnt/ground-data/prep_HAO_full/*pB*.fits" --data_path_tB "/mnt/ground-data/prep_HAO_full/*tB*.fits" --path_to_save "/mnt/ground-data/training/HAO_pinn_allviewpoint" --train "config/train.yaml" --hyperparameters "config/hyperparams_hao.yaml"


python -m sunerf.sunerf --wandb_name "hao_pinn_twoviewpoints_debug_martin" --data_path_pB "/mnt/ground-data/prep_HAO_2view_background/*pB*.fits" --data_path_tB "/mnt/ground-data/prep_HAO_2view_background/*tB*.fits" --path_to_save "/mnt/ground-data/training/HAO_pinn_2viewpoint_background_debug" --train "config/train.yaml" --hyperparameters "config/hyperparams_hao.yaml"

python -m sunerf.sunerf --wandb_name "hao_pinn_2viewpoints_background_no_physics" --data_path_pB "/mnt/prep_HAO_2view_background/*pB*.fits" --data_path_tB "/mnt/prep_HAO_2view_background/*tB*.fits" --path_to_save "/mnt/training/HAO_pinn_hao_pinn_2viewpoints_background_no_physics" --train "config/train.yaml" --hyperparameters "config/hyperparams_hao.yaml"


# Training with background

# Prep all fits files
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/mnt/ground-data/**/*.fits" --output_path /mnt/prep-data/prep_HAO_2view_background --check_matching

# Prep background fits files
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/mnt/ground-data/data_fits/*_bang_0000_*/stepnum_005.fits" --output_path /mnt/prep-data/prep_HAO_2view_background --check_matching

python -m sunerf.sunerf --wandb_name "hao_pinn_2viewpoints_background_martin" --data_path_pB "/mnt/ground-data/prep_HAO_2view/*pB*.fits" --data_path_tB "/mnt/ground-data/prep_HAO_2view_background/*tB*.fits" --path_to_save "/mnt/ground-data/training/HAO_pinn_2viewpoints_backgrounds" --train "config/train.yaml" --hyperparameters "config/hyperparams_hao.yaml"

#####################################
#                                   #
#   Download Data Preset Commands   #
#                                   #
#####################################

# Download all viewpoints first into data_fits subdirectory, then download all 005.step files as well.
gsutil -m cp -R gs://fdl23_europe_helio_onground/ground-data/data_fits /mnt/ground-data/data_fits

# Download 2 Viewpoints only into data_fits directory, emulating L5 and Earth (60° Diff) /mnt/ground-data/data_fits
gsutil -m cp -R  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_340W_bang_0000_tB /mnt/ground-data/data_fits/
gsutil -m cp -R  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_340W_bang_0000_pB /mnt/ground-data/data_fits/

gsutil -m cp -R  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_280W_bang_0000_tB /mnt/ground-data/data_fits/
gsutil -m cp -R  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_280W_bang_0000_pB /mnt/ground-data/data_fits/


# Download Background data into /mnt/ground-data/data_fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_020W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_020W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_020W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_020W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_040W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_040W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_040W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_040W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_060W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_060W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_060W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_060W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_080W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_080W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_080W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_080W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_100W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_100W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_100W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_100W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_120W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_120W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_120W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_120W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_140W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_140W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_140W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_140W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_160W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_160W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_160W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_160W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_180W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_180W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_180W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_180W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_200W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_200W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_200W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_200W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_220W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_220W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_220W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_220W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_240W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_240W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_240W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_240W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_260W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_260W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_260W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_260W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_280W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_280W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_280W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_280W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_300W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_300W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_300W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_300W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_320W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_320W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_320W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_320W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_340W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_340W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_340W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_340W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_360W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_360W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_360W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/data_fits/dcmer_360W_bang_0000_pB/stepnum_005.fits


################
#              #
#  Prep Data   #
#              #
################

# Prep all data available - only ever download the data that we would like to use
# Prep_HAO_2view
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/mnt/ground-data/data_fits/**/*.fits" --output_path /mnt/prep-data/prep_HAO_2view --check_matching

# Prep_HAO_2view_background
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/mnt/ground-data/data_fits/**/*.fits" --output_path /mnt/prep-data/prep_HAO_2view_background --check_matching

# prep_HAO_allview
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/mnt/ground-data/data_fits/**/*.fits" --output_path /mnt/prep-data/prep_HAO_allview --check_matching

######################
#                    #
#   Running ICARUS   #
#                    #
######################
# Prep_HAO_2view
python -m sunerf.sunerf --wandb_name "hao_pinn_2view" --data_path_pB "/mnt/prep-data/prep_HAO_2view/*pB*.fits" --data_path_tB "/mnt/prep-data/prep_HAO_2view/*tB*.fits" --path_to_save "/mnt/training/HAO_pinn_2view" --train "config/train.yaml" --hyperparameters "config/hyperparams_hao.yaml"
# prep_HAO_2view_backgrounds
python -m sunerf.sunerf --wandb_name "hao_pinn_2view_background" --data_path_pB "/mnt/prep-data/prep_HAO_2view_background/*pB*.fits" --data_path_tB "/mnt/prep-data/prep_HAO_2view_background/*tB*.fits" --path_to_save "/mnt/training/HAO_pinn_2view_background" --train "config/train.yaml" --hyperparameters "config/hyperparams_hao.yaml"
# prep_HAO_allview
python -m sunerf.sunerf --wandb_name "hao_pinn_all" --data_path_pB "/mnt/prep-data/prep_HAO_allview/*pB*.fits" --data_path_tB "/mnt/prep-data/prep_HAO_allview/*tB*.fits" --path_to_save "/mnt/training/HAO_pinn_allview" --train "config/train.yaml" --hyperparameters "config/hyperparams_hao.yaml"
