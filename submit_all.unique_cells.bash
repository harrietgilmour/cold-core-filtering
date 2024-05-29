#!/bin/sh -l
#
# This script submits the jobs to run the unique_cells.py script
#
# Usage: submit_all.unique_cells.bash <year>
#
# For example: bash submit_all.unique_cells.bash 2005
#

# Check that the year has been provided
if [ $# -ne 1 ]; then
    echo "Usage: submit_all.unique_cells.bash <year>"
    exit 1
fi

# extract the year from the command line
year=$1
month=$2

# echo the year
echo "Finding unique cells in year: $year"

# Set up months
months=(01 02 03 04 05 06 07 08 09 10 11 12)

# set up the extractor script
EXTRACTOR="/data/users/hgilmour/cold-core-filtering/submit.unique_cells.sh"

# Find the unique cells for given year
# base directory is the directory where the tracks are stored
# in format tracks_yyyy_mm.h5

base_dir="/project/cssp_brazil/mcs_tracking_HG/CPM_HINDCAST_TRACKS/init_tracks_cpm"


# Set up the output directory
OUTPUT_DIR="/project/cssp_brazil/mcs_tracking_HG/lotus_output/unique_cells"
mkdir -p $OUTPUT_DIR

# Loop over the months
for month in ${months[@]}; do
    
    echo $year
    echo $month

    # Find the tracks files for the given month
    tracks_file="tracks_${year}_${month}_INTERP.h5"
    # construct the tracks path
    tracks_path=${base_dir}/${tracks_file}

    # Set up the output files
    OUTPUT_FILE="$OUTPUT_DIR/all_unique_cells_INTERP.$year.$month.out"
    ERROR_FILE="$OUTPUT_DIR/all_unique_cells_INTERP.$year.$month.err"

    # submit the batch job
    sbatch --mem=1000 --ntasks=4 --time=15 --output=$OUTPUT_FILE --error=$ERROR_FILE $EXTRACTOR $tracks_path

done

