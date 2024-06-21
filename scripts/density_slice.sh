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

# 2 view
python -m sunerf.evaluation.density_cube_comparison --ckpt_path '/glade/work/rjarolim/sunerf-cme/background_v01/save_state.snf' --result_path '/glade/work/rjarolim/sunerf-cme/background_v01/comparison' --frames "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
python -m sunerf.evaluation.density_slice --ckpt_path '/glade/work/rjarolim/sunerf-cme/background_v01/save_state.snf' --result_path '/glade/work/rjarolim/sunerf-cme/background_v01/video'
python -m sunerf.evaluation.load_density_cube --ckpt_path '/glade/work/rjarolim/sunerf-cme/background_v01/save_state.snf' --result_path '/glade/work/rjarolim/sunerf-cme/background_v01/vtk'
python -m sunerf.evaluation.video_cme --ckpt_path '/glade/work/rjarolim/sunerf-cme/background_v01/save_state.snf' --result_path '/glade/work/rjarolim/sunerf-cme/background_v01/cme_video'

# 2 view no physics
python -m sunerf.evaluation.density_cube_comparison --ckpt_path '/glade/work/rjarolim/sunerf-cme/2view_no_physics_v01/save_state.snf' --result_path '/glade/work/rjarolim/sunerf-cme/2view_no_physics_v01/comparison' --frames "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
python -m sunerf.evaluation.density_slice --ckpt_path '/glade/work/rjarolim/sunerf-cme/2view_no_physics_v01/save_state.snf' --result_path '/glade/work/rjarolim/sunerf-cme/2view_no_physics_v01/video'


# 5 view
python -i -m sunerf.evaluation.density_cube_comparison --ckpt_path '/glade/work/rjarolim/sunerf-cme/5view_v01/save_state.snf' --result_path '/glade/work/rjarolim/sunerf-cme/5view_v01/comparison' --frames "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
python -m sunerf.evaluation.density_slice --ckpt_path '/glade/work/rjarolim/sunerf-cme/5view_v01/save_state.snf' --result_path '/glade/work/rjarolim/sunerf-cme/5view_v01/video'
python -m sunerf.evaluation.video_cme --ckpt_path '/glade/work/rjarolim/sunerf-cme/5view_v01/save_state.snf' --result_path '/glade/work/rjarolim/sunerf-cme/5view_v01/cme_video'
python -m sunerf.evaluation.load_density_cube --ckpt_path '/glade/work/rjarolim/sunerf-cme/5view_v01/save_state.snf' --result_path '/glade/work/rjarolim/sunerf-cme/5view_v01/vtk'

# all view
python -m sunerf.evaluation.density_cube_comparison --ckpt_path '/glade/work/rjarolim/sunerf-cme/all_v04/save_state.snf' --result_path '/glade/work/rjarolim/sunerf-cme/all_v04/comparison' --frames "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
python -m sunerf.evaluation.density_slice --ckpt_path '/glade/work/rjarolim/sunerf-cme/all_v04/save_state.snf' --result_path '/glade/work/rjarolim/sunerf-cme/all_v04/video'


# 1 view
python -m sunerf.evaluation.density_cube_comparison --ckpt_path '/glade/work/rjarolim/sunerf-cme/1view_v02/save_state.snf' --result_path '/glade/work/rjarolim/sunerf-cme/1view_v02/comparison' --frames "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
python -m sunerf.evaluation.density_slice --ckpt_path '/glade/work/rjarolim/sunerf-cme/1view_min_energy_v01/save_state.snf' --result_path '/glade/work/rjarolim/sunerf-cme/1view_min_energy_v01/video'
python -i -m sunerf.evaluation.load_density_cube --ckpt_path '/glade/work/rjarolim/sunerf-cme/1view_min_energy_v01/save_state.snf' --result_path '/glade/work/rjarolim/sunerf-cme/1view_min_energy_v01/vtk'


# others
python -m sunerf.evaluation.density_cube_comparison --ckpt_path '/glade/work/rjarolim/sunerf-cme/all_v04/save_state.snf' --result_path '/glade/work/rjarolim/sunerf-cme/all_v04/comparison' --frames "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
python -m sunerf.evaluation.density_cube_comparison --ckpt_path '/glade/work/rjarolim/sunerf-cme/5view_v01/save_state.snf' --result_path '/glade/work/rjarolim/sunerf-cme/5view_v01/comparison' --frames "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
python -m sunerf.evaluation.density_cube_comparison --ckpt_path '/glade/work/rjarolim/sunerf-cme/background_v02/save_state.snf' --result_path '/glade/work/rjarolim/sunerf-cme/background_v02/comparison' --frames "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
python -m sunerf.evaluation.density_cube_comparison --ckpt_path '/glade/work/rjarolim/sunerf-cme/2view_180_v01/save_state.snf' --result_path '/glade/work/rjarolim/sunerf-cme/2view_180_v01/comparison' --frames "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
python -m sunerf.evaluation.density_cube_comparison --ckpt_path '/glade/work/rjarolim/sunerf-cme/2view_270_v01/save_state.snf' --result_path '/glade/work/rjarolim/sunerf-cme/2view_270_v01/comparison' --frames "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
python -m sunerf.evaluation.density_cube_comparison --ckpt_path '/glade/work/rjarolim/sunerf-cme/2view_90_v01/save_state.snf' --result_path '/glade/work/rjarolim/sunerf-cme/2view_90_v01/comparison' --frames "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
python -m sunerf.evaluation.density_cube_comparison --ckpt_path '/glade/work/rjarolim/sunerf-cme/1view_v01/save_state.snf' --result_path '/glade/work/rjarolim/sunerf-cme/1view_v01/comparison' --frames "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
python -m sunerf.evaluation.density_cube_comparison --ckpt_path '/glade/work/rjarolim/sunerf-cme/2view_no_physics_v01/save_state.snf' --result_path '/glade/work/rjarolim/sunerf-cme/2view_no_physics_v01/comparison' --frames "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
python -m sunerf.evaluation.density_cube_comparison --ckpt_path '/glade/work/rjarolim/sunerf-cme/5view_no_physics_v01/save_state.snf' --result_path '/glade/work/rjarolim/sunerf-cme/5view_no_physics_v01/comparison' --frames "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"


