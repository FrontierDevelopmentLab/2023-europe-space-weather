#!/bin/bash -l

#PBS -N sunerf-cme-all
#PBS -A P22100000
#PBS -q casper
#PBS -l select=1:ncpus=16:ngpus=4:mem=256gb
#PBS -l walltime=12:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/sunerf-cme

python -i -m sunerf.evaluation.density_slice
