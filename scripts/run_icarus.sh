# Training from scratch
# init workspace

gsutil -m cp -R gs://fdl23_europe_helio_onground/ground-data/data_fits /mnt/ground-data/data_fits

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
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/mnt/ground-data/dcmer_340W_bang_0000_pB/*.fits" --output_path "/mnt/prep-data/prep_HAO_2view"
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/mnt/ground-data/dcmer_340W_bang_0000_tB/*.fits" --output_path "/mnt/prep-data/prep_HAO_2view"
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/mnt/ground-data/dcmer_280W_bang_0000_pB/*.fits" --output_path "/mnt/prep-data/prep_HAO_2view"
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/mnt/ground-data/dcmer_280W_bang_0000_tB/*.fits" --output_path "/mnt/prep-data/prep_HAO_2view"
# python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/mnt/ground-data/data_fits/**/*.fits" --output_path /mnt/prep-data/prep_HAO_all --check_matching
# full training
python -m sunerf.sunerf --wandb_name "hao_pinn_2viewpoints" --data_path_pB "/mnt/ground-data/prep_HAO/*pB*.fits" --data_path_tB "/mnt/ground-data/prep_HAO/*tB*.fits" --path_to_save "/mnt/ground-data/training/HAO_pinn_2viewpoint_v3" --train "config/train.yaml" --hyperparameters "config/hyperparams_hao.yaml"
python -m sunerf.sunerf --wandb_name "hao_pinn_allviewpoints" --data_path_pB "/mnt/ground-data/prep_HAO_full/*pB*.fits" --data_path_tB "/mnt/ground-data/prep_HAO_full/*tB*.fits" --path_to_save "/mnt/ground-data/training/HAO_pinn_allviewpoint" --train "config/train.yaml" --hyperparameters "config/hyperparams_hao.yaml"

python -m sunerf.sunerf --wandb_name "hao_pinn_cr_2viewpoints_background_a26978f" --data_path_pB "/mnt/prep-data/prep_HAO_2view_background/*pB*.fits" --data_path_tB "/mnt/prep-data/prep_HAO_2view_background/*tB*.fits" --path_to_save "/mnt/training/HAO_pinn_cr_2viewpoint_background_a26978f" --train "config/train.yaml" --hyperparameters "config/hyperparams_hao.yaml"
python -m sunerf.sunerf --wandb_name "hao_pinn_cr_2view_a26978f" --data_path_pB "/mnt/prep-data/prep_HAO_2view/*pB*.fits" --data_path_tB "/mnt/prep-data/prep_HAO_2view/*tB*.fits" --path_to_save "/mnt/training/HAO_pinn_cr_2view_a26978f" --train "config/train.yaml" --hyperparameters "config/hyperparams_hao.yaml"
python -m sunerf.sunerf --wandb_name "hao_pinn_cr_allview_a26978f" --data_path_pB "/mnt/prep-data/prep_HAO_allview/*pB*.fits" --data_path_tB "/mnt/prep-data/prep_HAO_allview/*tB*.fits" --path_to_save "/mnt/training/HAO_pinn_cr_allview_a26978f" --train "config/train.yaml" --hyperparameters "config/hyperparams_hao.yaml"

python -m sunerf.sunerf --wandb_name "hao_pinn_cr_allview_a26978f_heliographic" --data_path_pB "/mnt/prep-data/prep_HAO_allview/*pB*.fits" --data_path_tB "/mnt/prep-data/prep_HAO_allview/*tB*.fits" --path_to_save "/mnt/training/HAO_pinn_cr_allview_a26978f_heliographic" --train "config/train.yaml" --hyperparameters "config/hyperparams_hao.yaml"
