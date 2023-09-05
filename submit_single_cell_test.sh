#!/bin/bash
#SBATCH --mem=100000
#SBATCH --ntasks=4
#SBATCH --time=20

#Extract args from command line
mask_file=$1
precip_file=$2
tracks_file=$3
tb_file=$4
w_file=$5
cell=$6

echo "$mask_file"
echo "$precip_file"
echo "$tracks_file"
echo "$tb_file"
echo "$w_file"
echo "$cell"

python single_cell_loop.py ${mask_file} ${precip_file} ${tracks_file} ${tb_file} ${w_file} ${cell}
