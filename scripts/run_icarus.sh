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
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/mnt/ground-data/data_fits/dcmer_340W_bang_0000_pB/*.fits"
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/mnt/ground-data/data_fits/dcmer_340W_bang_0000_tB/*.fits"
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/mnt/ground-data/data_fits/dcmer_280W_bang_0000_pB/*.fits"
python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/mnt/ground-data/data_fits/dcmer_280W_bang_0000_tB/*.fits"
# python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/mnt/ground-data/data_fits/**/*.fits" --output_path /mnt/ground-data/prep_HAO_all
# full training
python -m sunerf.sunerf --wandb_name "hao_pinn_2viewpoints" --data_path_pB "/mnt/ground-data/prep_HAO/*pB*.fits" --data_path_tB "/mnt/ground-data/prep_HAO/*tB*.fits" --path_to_save "/mnt/ground-data/training/HAO_pinn_2viewpoint_v3" --train "config/train.yaml" --hyperparameters "config/hyperparams_hao.yaml"
python -m sunerf.sunerf --wandb_name "hao_pinn_allviewpoints" --data_path_pB "/mnt/ground-data/prep_HAO_full/*pB*.fits" --data_path_tB "/mnt/ground-data/prep_HAO_full/*tB*.fits" --path_to_save "/mnt/ground-data/training/HAO_pinn_allviewpoint" --train "config/train.yaml" --hyperparameters "config/hyperparams_hao.yaml"
