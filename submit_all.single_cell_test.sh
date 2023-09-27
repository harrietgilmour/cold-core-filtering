#!/bin/sh -l
#
# This script submits the jobs to run the single_cell_loop.py script
#
# Usage: bash submit_all.single_cell_test.sh <year>
#

# Check that the correct no of args has been passed
if [ $# -ne 1 ]; then
    echo "Usage: submit_all.single_cell_test.bash <year>"
    exit 1
fi

# Extract the year and month from the command line
year=$1

echo $year

# load the txt file
txt_file_path="/data/users/hgilmour/cold-core-filtering/unique_cell_files"
txt_file_name="unique_cells_${year}.txt"

# form the file path
txtfile=${txt_file_path}/${txt_file_name}

echo $txtfile

# Check that this file exists
if [ ! -f $txtfile ]; then
    echo "File not found!"
    exit 1
fi

# Set up the extractor script 
EXTRACTOR="/data/users/hgilmour/cold-core-filtering/submit_single_cell_test.sh"

# Set the output directory for the lOTUS OUTPUT
# Set up the output directory
OUTPUT_DIR="/data/users/hgilmour/cold-core-filtering/lotus_output/single_cell_test"
mkdir -p $OUTPUT_DIR

# We want to extract the array of values from the txtfile
unique_values_array=$(cat $txtfile)

# check that this array looks good
echo "array of unique values: ${unique_values_array}"

# set up the mask, precip and tracks file
mask_dir="/data/users/hgilmour/initial_tracks/tobac_initial_tracks/segmentation"
mask_file="segmentation_yearly_${year}.nc"

precip_dir="/scratch/hgilmour/total_precip"
precip_file="total_precip_${year}.nc"

tracks_dir="/data/users/hgilmour/initial_tracks/tobac_initial_tracks/tracking"
tracks_file="tracks_${year}.h5"

tb_dir="/data/users/hgilmour/tb"
tb_file="tb_${year}.nc"

w_dir="/scratch/hgilmour/omega_new"
w_file="omega_${year}.nc"

# form the file paths
mask=${mask_dir}/${mask_file}
precip=${precip_dir}/${precip_file}
tracks=${tracks_dir}/${tracks_file}
tb=${tb_dir}/${tb_file}
vert_vel=${w_dir}/${w_file}

# loop over the cells
for cell in ${unique_values_array[@]}; do

    echo $cell
    echo $mask
    echo $precip
    echo $tracks
    echo $vert_vel
    echo $tb

    # FIND THE SYNTAX FOR SBBATCH ON MET OFFICE PAGE

    # Set up the output files
    OUTPUT_FILE="$OUTPUT_DIR/all_single_cells_test.$year.$month.$cell.out"
    ERROR_FILE="$OUTPUT_DIR/all_single_cells_test.$year.$month.$cell.err"

    sbatch --mem=100000 --ntasks=4 --time=20 --output=$OUTPUT_FILE --error=$ERROR_FILE $EXTRACTOR $mask $precip $tracks $tb $vert_vel $cell

done



