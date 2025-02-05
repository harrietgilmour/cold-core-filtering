#!/bin/sh -l
#
# This script submits the jobs to run the single_cell_loop.py script
#
# Usage: bash submit_all.single_cell_test.sh <year> <month>
#

# Check that the correct no of args has been passed
if [ $# -ne 2 ]; then
    echo "Usage: submit_all.single_cell_test.bash <year> <month>"
    exit 1
fi

# Extract the year and month from the command line
year=$1
month=$2

echo $year
echo $month

# load the txt file
txt_file_path="/project/cssp_brazil/mcs_tracking_HG/CPM_F_TRACKS/unique_cells/orig_threshold"
txt_file_name="unique_cells_${year}_${month}.txt"

# form the file path
txtfile=${txt_file_path}/${txt_file_name}

echo $txtfile

# Check that this file exists
if [ ! -f $txtfile ]; then
    echo "File not found!"
    exit 1
fi

# Set up the extractor script 
EXTRACTOR="/home/h03/hgilmour/cold-core-filtering/submit_single_cell_test_F.sh"

# Set the output directory for the lOTUS OUTPUT
# Set up the output directory
OUTPUT_DIR="/project/cssp_brazil/mcs_tracking_HG/lotus_output/filter_F"
mkdir -p $OUTPUT_DIR

# We want to extract the array of values from the txtfile
unique_values_array=$(cat $txtfile)

# check that this array looks good
echo "array of unique values: ${unique_values_array}"

# set up the mask, precip and tracks file
mask_dir="/project/cssp_brazil/mcs_tracking_HG/CPM_F_TRACKS/segmentation/orig_threshold"
mask_file="segmentation_yearly_${year}.nc" # a year because these ad the tracks were done in yearly chunks so frame numbers are also based on year

precip_dir="/project/cssp_brazil/mcs_tracking_HG/data/cpm_FUTURE/total_precip/annual_files"
precip_file="total_precip_${year}.nc"

tracks_dir="/project/cssp_brazil/mcs_tracking_HG/CPM_F_TRACKS/init_tracks/orig_threshold"
tracks_file="tracks_${year}_${month}.h5"

tb_dir="/project/cssp_brazil/mcs_tracking_HG/data/cpm_FUTURE/tb/annual_files_hrly"
tb_file="tb_${year}.nc"

w_dir="/project/cssp_brazil/mcs_tracking_HG/data/cpm_FUTURE/omega"
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


    # Set up the output files
    OUTPUT_FILE="$OUTPUT_DIR/all_single_cells_test_F.$year.$month.$cell.out"
    ERROR_FILE="$OUTPUT_DIR/all_single_cells_test_F.$year.$month.$cell.err"

    echo $OUTPUT_FILE
    echo $ERROR_FILE

    sbatch --mem=5000 --ntasks=2 --time=8 --output=$OUTPUT_FILE --error=$ERROR_FILE $EXTRACTOR $mask $precip $tracks $tb $vert_vel $cell

done



