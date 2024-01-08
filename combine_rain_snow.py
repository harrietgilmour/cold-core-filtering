# Python script to combine rain and snow from the CPM PD simulation into a total precip variable
#
# <USAGE> python combine_rain_snow.py <RAIN_FILE> <SNOW_FILE>
#
# <EXAMPLE> python unique_cells.py /project/cssp_brazil/mcs_tracking_HG/init_tracks_obs/tracks_2006_01.h5
#


# Import local packages
import os
import sys
import glob

# Import third party packages
import numpy as np
import xarray as xr

# Import and set up warnings
import warnings
warnings.filterwarnings('ignore')


# Write a function which will check the number of arguements passed
def check_no_args(args):
    """Check the number of arguements passed"""
    if len(args) != 3:
        print('Incorrect number of arguements')
        print('Usage: python combine_rain_snow.py <RAIN_FILE> <SNOW_FILE>')
        print('Example: python combine_rain_snow.py <RAIN_FILE> <SNOW_FILE>')
        sys.exit(1)

# Write a function which loads the file
def open_datasets(rain_file, snow_file):
    """Load specified files"""

    #Load rain file
    rain = xr.open_dataset(rain_file)
    rain = rain.stratiform_rainfall_flux

    #Load snow file
    snow = xr.open_dataset(snow_file)
    snow = snow.stratiform_snowfall_flux 

    return rain, snow


#Define the main function / filerting loop:
def main():
    """Main function."""

    # First extract the arguements:
    rain_file = str(sys.argv[1])
    snow_file = str(sys.argv[2])

    #check the number of arguements
    check_no_args(sys.argv)

    #find the year of the file
    filename = os.path.basename(rain_file)
    print("Type of filename:", type(filename))
    print("Filename:", filename)
    filename_without_extension = os.path.splitext(filename)
    #print("Type of filename_without_extension:", type(filename_without_extension))
    print(filename_without_extension)
    filename = filename.replace(".", "_")
    segments = filename.split("_")
    print(segments)
    #segments = segments.split("_")
    #print(segments)
    year = segments[3]
    print("year:", year)
    month = segments[2]
    print("month:", month)

    #first open the tracks dataset for 1 month
    rain, snow = open_datasets(rain_file, snow_file)

    total_precip = rain + snow

    print("Shape of total_precip:", total_precip.shape)

    savepath = '/scratch/hgilmour/cpm_PD/total_precip/total_precip_{}_{}.nc'.format(month, year)
    print("Savepath:", savepath)

    total_precip.to_netcdf(savepath, mode='w')

#Run the main function
if __name__ == "__main__":
    main()

