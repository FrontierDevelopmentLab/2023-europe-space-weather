# Training from scratch
# init workspace

# convert data (pre-training with center crop)
python -m sunerf.prep.prep_hao
python -m sunerf.prep.prep_psi_cor --psi_path "/mnt/ground-data/PSI/pb_raw/*.fits"  --output_path "/mnt/ground-data/prep_PSI/pb_raw"
python -m sunerf.prep.prep_psi_cor --psi_path "/mnt/ground-data/PSI/b_raw/*.fits"  --output_path "/mnt/ground-data/prep_PSI/b_raw"
# full training PSI
python -m sunerf.sunerf --wandb_name "psi" --data_path_pB "/mnt/ground-data/prep_PSI/pb_raw/*.fits" --data_path_tB "/mnt/ground-data/prep_PSI/b_raw/*.fits" --path_to_save "/mnt/ground-data/training/PSI_v1" --train "config/train.yaml" --hyperparameters "config/hyperparams_icarus.yaml"

