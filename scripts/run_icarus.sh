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


# python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/mnt/ground-data/**/*.fits" --output_path /mnt/ground-data/prep_HAO_2view_background --check_matching
# full training

python -m sunerf.sunerf --wandb_name "hao_pinn_2viewpoints" --data_path_pB "/mnt/ground-data/prep_HAO/*pB*.fits" --data_path_tB "/mnt/ground-data/prep_HAO/*tB*.fits" --path_to_save "/mnt/ground-data/training/HAO_pinn_2viewpoint_v3" --train "config/train.yaml" --hyperparameters "config/hyperparams_hao.yaml"
python -m sunerf.sunerf --wandb_name "hao_pinn_allviewpoints" --data_path_pB "/mnt/ground-data/prep_HAO_full/*pB*.fits" --data_path_tB "/mnt/ground-data/prep_HAO_full/*tB*.fits" --path_to_save "/mnt/ground-data/training/HAO_pinn_allviewpoint" --train "config/train.yaml" --hyperparameters "config/hyperparams_hao.yaml"


python -m sunerf.sunerf --wandb_name "hao_pinn_twoviewpoints_debug_martin" --data_path_pB "/mnt/ground-data/prep_HAO_2view_background/*pB*.fits" --data_path_tB "/mnt/ground-data/prep_HAO_2view_background/*tB*.fits" --path_to_save "/mnt/ground-data/training/HAO_pinn_2viewpoint_background_debug" --train "config/train.yaml" --hyperparameters "config/hyperparams_hao.yaml"

python -m sunerf.sunerf --wandb_name "hao_pinn_twoviewpoints_heliographic" --data_path_pB "/mnt/ground-data/prep_HAO_2view_background/*pB*.fits" --data_path_tB "/mnt/ground-data/prep_HAO_2view_background/*tB*.fits" --path_to_save "/mnt/ground-data/training/HAO_pinn_2viewpoint_background_heliographic" --train "config/train.yaml" --hyperparameters "config/hyperparams_hao.yaml"


# Training with background


python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/mnt/ground-data/**/*.fits" --output_path /mnt/prep_HAO_2view_background --check_matching

python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/mnt/ground-data/data_fits/*_bang_0000_*/stepnum_005.fits" --output_path /mnt/ground-data/prep_HAO_2view_background --check_matching

python -m sunerf.sunerf --wandb_name "hao_pinn_2viewpoints_background_martin" --data_path_pB "/mnt/ground-data/prep_HAO_2view/*pB*.fits" --data_path_tB "/mnt/ground-data/prep_HAO_2view_background/*tB*.fits" --path_to_save "/mnt/ground-data/training/HAO_pinn_2viewpoints_backgrounds" --train "config/train.yaml" --hyperparameters "config/hyperparams_hao.yaml"

gsutil -m cp -R  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_340W_bang_0000_tB /mnt/ground-data/
gsutil -m cp -R  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_340W_bang_0000_pB /mnt/ground-data/

gsutil -m cp -R  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_280W_bang_0000_tB /mnt/ground-data/
gsutil -m cp -R  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_280W_bang_0000_pB /mnt/ground-data/

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_020W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/dcmer_020W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_020W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/dcmer_020W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_040W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/dcmer_040W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_040W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/dcmer_040W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_060W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/dcmer_060W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_060W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/dcmer_060W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_080W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/dcmer_080W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_080W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/dcmer_080W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_100W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/dcmer_100W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_100W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/dcmer_100W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_120W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/dcmer_120W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_120W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/dcmer_120W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_140W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/dcmer_140W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_140W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/dcmer_140W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_160W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/dcmer_160W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_160W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/dcmer_160W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_180W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/dcmer_180W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_180W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/dcmer_180W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_200W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/dcmer_200W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_200W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/dcmer_200W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_220W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/dcmer_220W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_220W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/dcmer_220W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_240W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/dcmer_240W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_240W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/dcmer_240W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_260W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/dcmer_260W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_260W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/dcmer_260W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_280W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/dcmer_280W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_280W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/dcmer_280W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_300W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/dcmer_300W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_300W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/dcmer_300W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_320W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/dcmer_320W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_320W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/dcmer_320W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_340W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/dcmer_340W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_340W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/dcmer_340W_bang_0000_pB/stepnum_005.fits

gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_360W_bang_0000_tB/stepnum_005.fits /mnt/ground-data/dcmer_360W_bang_0000_tB/stepnum_005.fits
gsutil -m cp  gs://fdl23_europe_helio_onground/ground-data/data_fits/dcmer_360W_bang_0000_pB/stepnum_005.fits /mnt/ground-data/dcmer_360W_bang_0000_pB/stepnum_005.fits

