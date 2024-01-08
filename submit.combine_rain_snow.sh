#!/bin/bash
#SBATCH --mem=100000
#SBATCH --ntasks=4
#SBATCH --time=45

#Extract args from command line
rain_file=$1
snow_file=$2

# Print the tracks file
echo "$rain_file"
echo "$snow_file"

# Run the unique_cells.py script
python combine_rain_snow.py ${rain_file} ${snow_file}