# -*- coding: utf-8 -*-
#--  Kasper Tølløse and Swapan Mallick
#--  5 August 2025
#-
import os
import sys
import subprocess
import shutil
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import zarr
from process_data_ERA5 import process_era5_data



# Parse arguments from command line
params_file = sys.argv[1]             # parameter list file
DTG = sys.argv[2]                     # Input Date (year and month)
DTSTR = int(sys.argv[3])              # Input Date (start date)
DTEND = int(sys.argv[4])              # Input Date (end date)
YY, MM = DTG[:4], DTG[4:6]



DATASET_NAME = "ERA5"


# Modify the input and output directories
SCRPDIR = "/home/dnk8136/Uncertainty_Quantification/CREATE_TRAINING_DATA"  # directory of conversion script
TMPDIR = f"{SCRPDIR}/tmp_WORKING_DIR"                                      # working directory
OUTDIR = f"/ec/res4/scratch/dnk8136/DDPM_DATA/"                            # path to output


INPUT2 = f"/scratch/fasg/CARRA2/uncert_est/"                               # path to ERA5 netcdf files


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

    # for HH in ['00']:  # loop over hours
    for HH in ['00', '06', '12', '18']:  # loop over hours

        timestamp = f"{DTG+DDN+HH}"

        # If the file already exists, check for the date
        if os.path.exists(f"{OUTDIR}/{DATASET_NAME}.zarr" ):
            existing_ds = xr.open_zarr(f"{OUTDIR}/{DATASET_NAME}.zarr" )
            # check if variable exists in dataset already
            if timestamp in existing_ds['time'].values:
                print(f"Date {timestamp} already processed. Skipping.", flush=True)
                continue


        # initialize empty dataset to contain all parameters
        ds_all_params = xr.Dataset()

        for param in list_params:  # loop over parameters

            # create and change to temp dir
            os.makedirs(TMPDIR, exist_ok=True)
            os.chdir(TMPDIR)
            

            # process CARRA2 data (for param)
            ds_param = process_era5_data(INPUT2,
                                         param,
                                         YY, MM, DDN, HH)
            print("carra2", ds_param, flush=True)


            # merge single parameter datasets with all-params dataset (for both carra2 and era5)
            ds_all_params = xr.merge([ds_all_params, ds_param])
            print("dataset, all parameters:", ds_all_params, flush=True)


            # move figures---
            OUTPUT=f"{OUTDIR}/PLOTS/{DTG}/{param}"
            os.makedirs(OUTPUT, exist_ok=True)
            for png1 in ['SD', 'EnsMEAN']:
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
