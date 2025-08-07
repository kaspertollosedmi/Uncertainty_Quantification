#!/usr/bin/env bash
#SBATCH --output=log.out
#SBATCH --error=log.out
#SBATCH --job-name=CONV2ZARR
#SBATCH --qos=nf

module load python3


# Define your variables
params_list="$1"
DTG="$2"
DTSTR="$3"
DTEND="$4"
DATASET="$5"


# # Define your variables
# params_list="sp,t2m"
# # params_list="list_examples.txt"
# # params_list="list_params_all.txt"
# DTG="202201"
# DTSTR=5
# DTEND=30
# # DATASET="CARRA2"
# DATASET="ERA5"


# run data conversion
python3 CONVERT2ZARR_$DATASET.py "$params_list" "$DTG" "$DTSTR" "$DTEND"


# script for inspecting file
# python3 INSPECT_ZARR_ARCHIVE.py
