#!/bin/bash

link="https://stereo-ssc.nascom.nasa.gov/instruments/software/secchi/utils/icer/"
filenames=("icomp" "idecomp" "idecomp.linux" "idecompx" "idecompx.ppc")

output_dir="tools/"

for filename in "${filenames[@]}"
do
    wget -P ${output_dir} "${link}${filename}"
done
