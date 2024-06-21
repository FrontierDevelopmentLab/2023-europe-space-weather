#!/bin/bash -l

#PBS -N sunerf-cme-all
#PBS -A P22100000
#PBS -q preempt
#PBS -l select=1:ncpus=4:ngpus=2:mem=64gb
#PBS -l walltime=12:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/sunerf-cme

# others
python -m sunerf.evaluation.density_cube_comparison --ckpt_path '/glade/work/rjarolim/sunerf-cme/2view_180_v01/save_state.snf' --result_path '/glade/work/rjarolim/sunerf-cme/2view_180_v01/comparison' --frames "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
python -m sunerf.evaluation.density_cube_comparison --ckpt_path '/glade/work/rjarolim/sunerf-cme/2view_270_v01/save_state.snf' --result_path '/glade/work/rjarolim/sunerf-cme/2view_270_v01/comparison' --frames "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
python -m sunerf.evaluation.density_cube_comparison --ckpt_path '/glade/work/rjarolim/sunerf-cme/2view_90_v01/save_state.snf' --result_path '/glade/work/rjarolim/sunerf-cme/2view_90_v01/comparison' --frames "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
python -m sunerf.evaluation.density_cube_comparison --ckpt_path '/glade/work/rjarolim/sunerf-cme/all_v04/save_state.snf' --result_path '/glade/work/rjarolim/sunerf-cme/all_v04/comparison' --frames "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"