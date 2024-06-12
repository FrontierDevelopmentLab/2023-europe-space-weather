#!/bin/bash -l

#PBS -N sunerf-cme
#PBS -A P22100000
#PBS -q preempt
#PBS -l select=1:ncpus=16:ngpus=4:mem=256gb
#PBS -l walltime=24:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/sunerf-cme

# Prep 2view
#python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_320W_bang_0000_pB/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_2view"
#python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_320W_bang_0000_tB/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_2view"
#python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_020W_bang_0000_pB/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_2view"
#python -m sunerf.prep.prep_hao --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_020W_bang_0000_tB/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_2view"

python -i -m sunerf.run --config "config/hao_2view.yaml"
