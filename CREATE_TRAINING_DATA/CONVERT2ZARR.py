# -*- coding: utf-8 -*-
#--  Kasper Tølløse and Swapan Mallick
#--  1 August 2025
#-
import os
import sys
import subprocess
import shutil
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import zarr
from process_data import process_carra2_data, process_era5_data


DATASET_NAME = "TRAINING_DATA"
# DATASET_NAME = "CARRA2"
# DATASET_NAME = "ERA5"


# Modify the input and output directories
SCRPDIR = "/home/dnk8136/Uncertainty_Quantification/CREATE_TRAINING_DATA"  # directory of conversion script
TMPDIR = f"{SCRPDIR}/tmp_WORKING_DIR"                                      # working directory
INPUT1 = f"/ec/res4/scratch/swe4281/DDPM_DATA/KESP/NCFILES/"               # path to CARRA2 netcdf files
INPUT2 = f"/ec/res4/scratch/dnk8136/DDPM_DATA/ERA5/"                       # path to ERA5 netcdf files
# INPUT1 = f"/ec/res4/scratch/swe4281/DDPM_DATA/CARRA2_SUMMER/NCFILES/"      # path to CARRA2 netcdf files
# INPUT2 = f"/scratch/fasg/CARRA2/uncert_est/"                               # path to ERA5 netcdf files
OUTDIR = f"/ec/res4/scratch/dnk8136/DDPM_DATA/"                  # path to output
plot_png = True                                                            # optional plotting (mainly to verify that the algorithm works)

# Input Date Configuration
DTG = "202203"
DTSTR, DTEND = 15, 15
YY, MM = DTG[:4], DTG[4:6]


# set down sampling method for CARRA2 data
downsampling_option = None  # None/"averaging"/"subsampling"
downsampling_factor = 1   # e.g. 12 -> 2869//12 = 239


# interpolate era5 data to the carra2 grid?
interpolation_method = None  # None/"nearest"/"linear"
if downsampling_option is None: interpolation_method = None


# parameter list
params_file = 'list_examples.txt'
# params_file = 'list_all.txt'


# Load Parameters from External File
if not os.path.exists(params_file):
    raise FileNotFoundError(f"Parameter file '{params_file}' not found.")
with open(params_file, 'r') as f:
    list_params = [line.strip() for line in f if line.strip()]

# create directories
os.makedirs(f"{OUTDIR}", exist_ok=True)
os.makedirs(f"{OUTDIR}/PLOTS", exist_ok=True)


# Main Processing Loop
for DD in range(DTSTR, DTEND+1):  # loop over days
    DDN = f"{DD:02}"

    for HH in ['00', '06', '12', '18']:  # loop over hours

        timestamp = f"{DTG+DDN+HH}"

        # If the file already exists, check for the date
        if os.path.exists(f"{OUTDIR}/{DATASET_NAME}.zarr" ):
            existing_ds = xr.open_zarr(f"{OUTDIR}/{DATASET_NAME}.zarr" )
            # check if variable exists in dataset already
            if timestamp in existing_ds['time'].values:
                print(f"Date {timestamp} already processed. Skipping.")
                continue

        # initialize empty dataset to contain all parameters
        ds_all_params = xr.Dataset()


        for param in list_params:  # loop over parameters

            # create and change to temp dir
            os.makedirs(TMPDIR, exist_ok=True)
            os.chdir(TMPDIR)
            

            # process CARRA2 data (for param)
            ds_param_carra2 = process_carra2_data(INPUT1,
                                                  OUTDIR,
                                                  param,
                                                  timestamp,
                                                  YY, MM, DDN, HH,
                                                  downsampling_option,
                                                  downsampling_factor,
                                                  plot_png)
            # print("carra2", ds_param_carra2, flush=True)


            # process ERA5 data (for param)
            ds_param_era5 = process_era5_data(INPUT2,
                                              OUTDIR,
                                              param,
                                              timestamp,
                                              YY, MM, DDN, HH,
                                              interpolation_method,
                                              (ds_param_carra2.Y, ds_param_carra2.X),   # carra2 dimensions
                                              plot_png)
            # print("era5", ds_param_era5, flush=True)


            # merge single parameter datasets with all-params dataset (for both carra2 and era5)
            ds_all_params = xr.merge([ds_all_params, ds_param_carra2])
            ds_all_params = xr.merge([ds_all_params, ds_param_era5])
            # print("dataset, all parameters:", ds_all_params, flush=True)


            if plot_png:
                # move figures---
                OUTPUT=f"{OUTDIR}/PLOTS/{DTG}/{param}"
                os.makedirs(OUTPUT, exist_ok=True)
                for png1 in ['SD', 'EnsMEAN']:
                    if os.path.exists(f"{png1}_carra2.png"):
                        file_out=f"{OUTPUT}/{png1}_{param}_{YY}{MM}{DDN}{HH}_carra2.png"
                        mvfile = ['mv', f"{png1}_carra2.png", f"{file_out}"]
                        subprocess.run(mvfile)
                    if os.path.exists(f"{png1}_era5.png"):
                        file_out=f"{OUTPUT}/{png1}_{param}_{YY}{MM}{DDN}{HH}_era5.png"
                        mvfile = ['mv', f"{png1}_era5.png", f"{file_out}"]
                        subprocess.run(mvfile)

            # delete TMPDIR---
            os.chdir(SCRPDIR)
            subprocess.run(["rm", "-rfd", TMPDIR], cwd=SCRPDIR)

        # write dataset to zarr archive
        if os.path.exists(f"{OUTDIR}/{DATASET_NAME}.zarr" ):
            # write to zarr
            print("\nfile '{DATASET_NAME}.zarr' exists. Data is appended to existing archive.", flush=True)
            ds_all_params.to_zarr(f"{OUTDIR}/{DATASET_NAME}.zarr" , mode='a', append_dim="time")
        else:
            print("\nfile '{DATASET_NAME}.zarr' does not exists. New archive is created.", flush=True)
            ds_all_params.to_zarr(f"{OUTDIR}/{DATASET_NAME}.zarr" , mode='w')
        print(f"Data for {timestamp} successfully written to zarr database.", flush=True)
        print("Current content of zarr archive:\n", xr.open_dataset(f"{OUTDIR}/{DATASET_NAME}.zarr"), flush=True)
    #  HH

print("Final content of zarr archive:\n", xr.open_dataset(f"{OUTDIR}/{DATASET_NAME}.zarr"), flush=True)
quit()
