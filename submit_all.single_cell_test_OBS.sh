#!/bin/sh -l
#
# This script submits the jobs to run the single_cell_loop.py script
#
# Usage: bash submit_all.single_cell_test_OBS.sh <year> <month>
#

# Check that the correct no of args has been passed
if [ $# -ne 2 ]; then
    echo "Usage: submit_all.single_cell_test_OBS.sh <year> <month>"
    exit 1
fi

# Extract the year and month from the command line
year=$1
month=$2

echo $year
echo $month

# load the txt file
#txt_file_path="/data/users/hgilmour/cold-core-filtering/unique_cell_files_obs"
txt_file_path="/project/cssp_brazil/mcs_tracking_HG/OBS_TRACKS/unique_cells/interp_version"
txt_file_name="unique_cells_${year}_${month}_INTERP.txt"

# form the file path
txtfile=${txt_file_path}/${txt_file_name}

echo $txtfile

# Check that this file exists
if [ ! -f $txtfile ]; then
    echo "File not found!"
    exit 1
fi

# Set up the extractor script 
EXTRACTOR="/data/users/hgilmour/cold-core-filtering/submit_single_cell_test_OBS.sh"

# Set the output directory for the lOTUS OUTPUT
# Set up the output directory
OUTPUT_DIR="/project/cssp_brazil/mcs_tracking_HG/lotus_output/obs_filter"
mkdir -p $OUTPUT_DIR

# We want to extract the array of values from the txtfile
unique_values_array=$(cat $txtfile)

# check that this array looks good
echo "array of unique values: ${unique_values_array}"

# set up the mask, precip and tracks file
mask_dir="/project/cssp_brazil/mcs_tracking_HG/OBS_TRACKS/segmentation_obs"
mask_file="segmentation_yearly_${year}_INTERP.nc" # a year because these ad the tracks were done in yearly chunks so frame numbers are also based on year

precip_dir="/scratch/hgilmour/obs/precip"
precip_file="precip_${year}.nc"

tracks_dir="/project/cssp_brazil/mcs_tracking_HG/OBS_TRACKS/init_tracks_obs/interp"
tracks_file="tracks_${year}_${month}_INTERP.h5"

tb_dir="/scratch/hgilmour/obs/tb/annual_files_hrly/regridded/interpolated"
tb_file="interp_regridded_tb_${year}.nc"


# form the file paths
mask=${mask_dir}/${mask_file}
precip=${precip_dir}/${precip_file}
tracks=${tracks_dir}/${tracks_file}
tb=${tb_dir}/${tb_file}

# loop over the cells
for cell in ${unique_values_array[@]}; do

    echo $cell
    echo $mask
    echo $precip
    echo $tracks
    echo $tb


    # Set up the output files
    OUTPUT_FILE="$OUTPUT_DIR/all_single_cells_test_INTERP.$year.$cell.out"
    ERROR_FILE="$OUTPUT_DIR/all_single_cells_test_INTERP.$year.$cell.err"

    sbatch --mem=5000 --ntasks=2 --time=3 --output=$OUTPUT_FILE --error=$ERROR_FILE $EXTRACTOR $mask $precip $tracks $tb $cell

done



