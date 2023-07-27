# Training from scratch
# init workspace
conda init
source ~/.bashrc
conda activate /mount/robert_jarolim/envs/fdl22v2
git config --global --add safe.directory /mount/robert_jarolim/4piuvsun2
wandb login 16f209a0af67d2365b17f1fc68567d3d908f76d5
cd /mount/robert_jarolim/4piuvsun2
# convert data (pre-training with center crop)
python -m s4pi.maps.prep.prep_sdo --sdo_file_path "/mount/nerf_data/sdo_2012_08/1h_193/*.fits" --output_path "/workspace/prep_2012_08/193" --center_crop True
python -m s4pi.maps.prep.prep_stereo --stereo_file_path "/mount/nerf_data/stereo_2012_08_converted_fov/195/*.fits" --output_path "/workspace/prep_2012_08/193" --center_crop True
# training step for 1 epoch
python -m s4pi.maps.sunerf --n_epochs 1 --data_path "/workspace/prep_2012_08/193/*" --path_to_save "/mount/robert_jarolim/results/fov_193_crop" --train "config/dgx_train.yaml" --hyperparameters "config/hyperparams.yaml"
# convert data (full training without center crop)
# clear previous data
rm -r /workspace/prep_2012_08/193
python -m s4pi.maps.prep.prep_sdo --sdo_file_path "/mount/nerf_data/sdo_2012_08/1h_193/*.fits" --output_path "/workspace/prep_2012_08/193" --center_crop False
python -m s4pi.maps.prep.prep_stereo --stereo_file_path "/mount/nerf_data/stereo_2012_08_converted_fov/195/*.fits" --output_path "/workspace/prep_2012_08/193" --center_crop False
# full training
python -m s4pi.maps.sunerf --wandb_name "193" --resume_from_checkpoint "/mount/robert_jarolim/results/fov_193_crop/final.ckpt" --data_path "/workspace/prep_2012_08/193/*.fits" --path_to_save "/mount/robert_jarolim/results/fov_193" --train "config/dgx_train.yaml" --hyperparameters "config/hyperparams.yaml"

