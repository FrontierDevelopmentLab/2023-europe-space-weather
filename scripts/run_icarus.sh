# Training from scratch
# init workspace

# convert data (pre-training with center crop)
python -m s4pi.maps.prep.prep_hao
# full training
python -m sunerf.sunerf --wandb_name "hao" --data_path_pB "/mnt/ground-data/prep_HAO/*pB*.fits" --data_path_tB "/mnt/ground-data/prep_HAO/*tB*.fits" --path_to_save "/mnt/ground-data/training/HAO_v1" --train "config/train.yaml" --hyperparameters "config/hyperparams_icarus.yaml"

