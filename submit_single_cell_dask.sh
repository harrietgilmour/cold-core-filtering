#!/bin/bash
#SBATCH --mem=200000
#SBATCH --ntasks=4
#SBATCH --time=1440
#SBATCH --qos=long
#SBATCH --output=/data/users/hgilmour/cold-core-filtering/lotus_output/single_cell_test
#SBATCH --error=/data/users/hgilmour/cold-core-filtering/lotus_output/single_cell_test

# Check that the correct no of args has been passed
if [ $# -ne 2 ]; then
    echo "Usage: submit_all.single_cell_test_1.bash <year> <month>"
    exit 1
fi

# Extract the year and month from the command line
year=$1
month=$2

echo $year
echo $month

python dask_single_cell_loop.py ${year} ${month}
