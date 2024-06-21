#!/bin/bash -l

#PBS -N sunerf-cme-all
#PBS -A P22100000
#PBS -q preempt
#PBS -l select=1:ncpus=16:ngpus=4:mem=256gb
#PBS -l walltime=12:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/sunerf-cme

# Prep allview
#python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_all"  --check_matching

# run allview
python -m sunerf.run --config "config/hao_all_no_physics.yaml"
