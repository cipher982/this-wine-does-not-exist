#!/bin/bash

BLUR_DIR="/home/drose/github/BlurDetection2/"
OUTPUT_PATH="/mnt/h/data/stylegan2_ada_outputs/blur_results.json"

for (( class=0; class <=15; class++ ))
do
cd $BLUR_DIR
python3 process.py \
 -i /mnt/h/data/stylegan2_ada_outputs/outputs_00011-9480_$class/ \
 -s $OUTPUT_PATH
done