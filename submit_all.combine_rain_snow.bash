#!/bin/sh -l
#
# This script submits the jobs to run the combine_rain_snow.py script
#
# Usage: submit_all.combine_rain_snow.bash <year>
#
# For example: bash submit_all.combine_rain_snow.bash 2005
#

# Check that the year has been provided
if [ $# -ne 1 ]; then
    echo "Usage: submit_all.combine_rain_snow.bash <year>"
    exit 1
fi

# extract the year from the command line
year=$1
month=$2

# echo the year
echo "Combining rain and snow in year: $year"

# Set up months
months=(01 02 03 04 05 06 07 08 09 10 11 12)

# set up the extractor script
EXTRACTOR="/data/users/hgilmour/cold-core-filtering/submit.combine_rain_snow.sh"

# Find the unique cells for given year
# base directory is the directory where the tracks are stored
# in format tracks_yyyy_mm.h5

rain_base_dir="/scratch/hgilmour/cpm_PD/precip"
snow_base_dir="/scratch/hgilmour/cpm_PD/snow"



# Set up the output directory
OUTPUT_DIR="/project/cssp_brazil/mcs_tracking_HG/lotus_output/combine_rain_snow"
mkdir -p $OUTPUT_DIR

# Loop over the months
for month in ${months[@]}; do
    
    echo $year
    echo $month

    # Find the precip files for the given month
    rain_file="precip_merge_${month}_${year}.nc"
    # construct the rain path
    rain_path=${rain_base_dir}/${year}/${rain_file}

    # Find the snow files for the given month
    snow_file="snow_merge_${month}_${year}.nc"
    # construct the snow path
    snow_path=${snow_base_dir}/${year}/${snow_file}

    # Set up the output files
    OUTPUT_FILE="$OUTPUT_DIR/combine_rain_snow.$year.$month.out"
    ERROR_FILE="$OUTPUT_DIR/combine_rain_snow.$year.$month.err"

    # submit the batch job
    sbatch --mem=100000 --ntasks=4 --time=45 --output=$OUTPUT_FILE --error=$ERROR_FILE $EXTRACTOR $rain_path $snow_path

done

