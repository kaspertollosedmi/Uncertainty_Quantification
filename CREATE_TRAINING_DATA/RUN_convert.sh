#!/usr/bin/env bash
#SBATCH --output=log.out
#SBATCH --error=log.out
#SBATCH --job-name=CONV2ZARR
#SBATCH --qos=nf

module load python3


# Define your variables
params_file="list_examples.txt"
DTG="202201"
DTSTR=5
DTEND=5
DATASET="CARRA2"
# DATASET="ERA5"

# run data conversion
python3 CONVERT2ZARR_$DATASET.py "$params_file" "$DTG" "$DTSTR" "$DTEND"


# script for inspecting file
# python3 INSPECT_ZARR_ARCHIVE.py
