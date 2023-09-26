#!/bin/bash
#SBATCH --mem=10000
#SBATCH --ntasks=4
#SBATCH --time=15

#Extract args from command line
tracks_file=$1

# Print the tracks file
echo "$tracks_file"

# Run the unique_cells.py script
python unique_cells.py ${tracks_file}
