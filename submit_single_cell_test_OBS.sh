#!/bin/bash
#SBATCH --mem=5000
#SBATCH --ntasks=2
#SBATCH --time=2

#Extract args from command line
mask_file=$1
precip_file=$2
tracks_file=$3
tb_file=$4
cell=$5

echo "$mask_file"
echo "$precip_file"
echo "$tracks_file"
echo "$tb_file"
echo "$cell"

python single_cell_loop_OBS.py ${mask_file} ${precip_file} ${tracks_file} ${tb_file} ${cell}