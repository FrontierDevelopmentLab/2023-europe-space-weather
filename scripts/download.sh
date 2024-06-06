#!/bin/bash -l

#PBS -N 13664
#PBS -A P22100000
#PBS -q preempt
#PBS -l select=1:ncpus=8:ngpus=2:mem=24gb
#PBS -l walltime=12:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/sunerf-cme
python3 -m sunerf.data.download_globus --download_dir "/glade/work/rjarolim/data/sunerf-cme/hao"