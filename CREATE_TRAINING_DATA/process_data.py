# -*- coding: utf-8 -*-
#--  Kasper Tølløse and Swapan Mallick
#--  1 August 2025
#-
import os
import sys
from pathlib import Path
import subprocess
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import zarr
from datetime import datetime


# list of FA names corresponding to variable names in era5 netcdf files
FA_name = {"t2m": "CLSTEMPERATURE",
           "q2m": "CLSHUMI.SPECIFIQ",
           "u10": "CLSVENT.ZONAL",
           "v10": "CLSVENT.MERIDIEN",
           "sp": "SURFPRESSION",
           "nlwrt": "SOMMRAYT.TERREST",
           "t950": "S051TEMPERATURE",
           "t900": "S046TEMPERATURE",
           "t700": "S033TEMPERATURE",
           "t500": "S024TEMPERATURE",
           "q950": "S051HUMI.SPECIFI",
           "q900": "S046HUMI.SPECIFI",
           "q700": "S033HUMI.SPECIFI",
           "q500": "S024HUMI.SPECIFI",
           "u950": "S051WIND.U.PHYS",
           "u900": "S046WIND.U.PHYS",
           "u700": "S033WIND.U.PHYS",
           "u500": "S024WIND.U.PHYS",
           "v950": "S051WIND.V.PHYS",
           "v900": "S046WIND.V.PHYS",
           "v700": "S033WIND.V.PHYS",
           "v500": "S024WIND.V.PHYS"}

# Variables that need scaling
scale_factors_carra2 = {
    'sp': 100.,'q950': 10000.,'q900': 10000.,'q700': 10000.,'q500': 10000.
}

scale_factors_era5 = {
    'sp': 0.01,'q950': 10000.,'q900': 10000.,'q700': 10000.,'q500': 10000.
}

# Plotting style
plt.rcParams.update({
    'font.size': 16,
    'axes.linewidth': 1.5,
    'font.family': 'serif'
})

# Define variable-specific plotting ranges
variable_ranges = {
    'sp': (500, 1170),
    'u10': (-30, 30),'u950': (-30, 30),'u900': (-30, 30),'u700': (-30, 30),'u500': (-30, 30),
    'v10': (-30, 30),'v950': (-30, 30),'v900': (-30, 30),'v700': (-30, 30),'v500': (-30, 30),
    't2m': (200, 300),'t950': (200, 300),'t900': (200, 300),'t700': (200, 300),'t500': (200, 300),
    'q950': (-30, 30),'q900': (-30, 30),'q700': (-30, 30),'q500': (-30, 30)
}


def process_carra2_data(PATH_TO_DATA, OUTDIR, VARIABLE, timestamp, YY, MM, DDN, HH, downsampling_option, downsampling_factor, plot_png):
    print(f"\nCARRA2 variable to be processed: {VARIABLE} ({FA_name[VARIABLE]}), for {timestamp}", flush=True)

    # first, create local links to all carra2 ensemble members
    for mem in range(10):  # Members from 0 to 9
            mem_IN = Path(f"{PATH_TO_DATA}/{YY}/{MM}/{DDN}/{HH}/mbr00{mem}/FC006")     # CARRA2ENDA
            # print(f'Processing Member {mem_IN}', flush=True)
            command = ['ln', '-sf', f"{mem_IN}/{FA_name[VARIABLE]}.nc", f"FILE{mem}.nc"]
            subprocess.run(command)
            
    # File names
    file_names = [
        'FILE0.nc', 'FILE1.nc', 'FILE2.nc', 'FILE3.nc', 'FILE4.nc', 
        'FILE5.nc', 'FILE6.nc', 'FILE7.nc', 'FILE8.nc', 'FILE9.nc'
    ]

    # Load data
    data = xr.open_mfdataset(file_names,
                             combine='nested',
                             concat_dim='ensemble'
                             )

    # Variables that need scaling
    if VARIABLE in scale_factors_carra2: data[FA_name[VARIABLE]] *= scale_factors_carra2[VARIABLE]

    # Compute statistics
    ensemble_mean = data[FA_name[VARIABLE]].mean(dim='ensemble')
    ensemble_std = data[FA_name[VARIABLE]].std(dim='ensemble')

    # apply downsampling to get CARRA2 ensemble statistics in coarser resolution
    if downsampling_option == "subsampling":
        # coarsen the data simply by subsampling
        ensemble_mean = ensemble_mean.sel(Y=data.Y[::downsampling_factor], X=data.X[::downsampling_factor])
        ensemble_std = ensemble_std.sel(Y=data.Y[::downsampling_factor], X=data.X[::downsampling_factor])
    elif downsampling_option == "averaging":
        # coarsen the data, i.e. calculate averages of subgrids to reduce dimensions
        ensemble_mean = ensemble_mean.coarsen(Y=downsampling_factor, X=downsampling_factor, boundary='trim').mean()
        ensemble_std = ensemble_std.coarsen(Y=downsampling_factor, X=downsampling_factor, boundary='trim').mean()

    # add coordinate information to data array 
    coord_dict = {'Y': ensemble_std.Y, 'X': ensemble_std.X}
    ensemble_std = xr.DataArray(ensemble_std, coords=coord_dict, dims=['Y', 'X'])
    ensemble_mean = xr.DataArray(ensemble_mean, coords=coord_dict, dims=['Y', 'X'])

    # Rechunk the DataArray to match the dimension of the grid
    if downsampling_option is not None:
        ensemble_std = ensemble_std.chunk({'Y': len(ensemble_std.Y), 'X': len(ensemble_std.X)})
        ensemble_mean = ensemble_mean.chunk({'Y': len(ensemble_mean.Y), 'X': len(ensemble_mean.X)})


    # Create new Dataset containing processed data
    ds_processed = xr.Dataset({VARIABLE+"_carra2_mean": ensemble_mean,
                               VARIABLE+"_carra2_std": ensemble_std})


    # expand dimension (add date, to be able to append data to exisitng zarr dataset)
    ds_processed = ds_processed.expand_dims({'time': [timestamp]})
    print(f"\nCARRA2 variable was successfully processed: {VARIABLE}, for {timestamp}\n", flush=True)


    # plot fields
    if plot_png:
        ens_mean = ds_processed[VARIABLE+"_carra2_mean"]
        ens_std = ds_processed[VARIABLE+"_carra2_std"]


        # Define plotting levels if available
        if VARIABLE in variable_ranges:
            min_level, max_level = variable_ranges[VARIABLE]
            levels = np.arange(np.floor(min_level), np.ceil(max_level) + 2, 2)
        else:
            min_level = float(np.floor(ens_mean.values.min()))
            max_level = float(np.ceil(ens_mean.values.max()))
            levels = np.linspace(min_level, max_level, num=10)


        vmax = 3
        if VARIABLE == "sp": vmax = 0.1

        # --------------------------- Save Plots --------------------------- #
        plot_sd(
            "Standard Deviation", 
            ens_std,
            "SD_carra2.png",
            # f"{OUTDIR}/PLOTS/SD_{VARIABLE}_{YY}{MM}{DDN}_carra2.png",
            f"{YY}{MM}{DDN}",
            vmin=0,
            vmax=vmax
            #vmax=np.ceil(ens_std.values.max()),
        )

        plot_ens_mean(
            "Ensemble Mean", 
            ens_mean,
            "EnsMEAN_carra2.png",
            # f"{OUTDIR}/PLOTS/EnsMEAN_{VARIABLE}_{YY}{MM}{DDN}_carra2.png",
            f"{YY}{MM}{DDN}",
            #vmin=np.floor(ens_mean.values.min()),
            #vmax=np.ceil(ens_mean.values.max()),
            levels
        )

        # ----------------------------------------------------------------- #

        print(f"png was generated for: {VARIABLE}, for {timestamp}\n", flush=True)

    return ds_processed



def process_era5_data(PATH_TO_DATA, OUTDIR, VARIABLE, timestamp, YY, MM, DDN, HH, interpolation_method, carra_dims, plot_png):
    DATE = timestamp[:-2]

    print(f"\nERA5 variable to be processed: {VARIABLE}, for {timestamp}", flush=True)

    # open dataset 
    if VARIABLE in ["t2m", "u10", "v10", "sp"]:   # surface variables
        data = xr.open_dataset(PATH_TO_DATA+f"/ERA5_EDA_SFC_{DATE}.nc")

        # Variables that need scaling
        if VARIABLE in scale_factors_era5:
            data[VARIABLE] *= scale_factors_era5[VARIABLE]

        # Compute statistics, and flip y axis
        ensemble_mean = data[VARIABLE].sel(time=datetime.strptime(timestamp, "%Y%m%d%H"), y=data.y[::-1]).mean(dim='number')
        ensemble_std = data[VARIABLE].sel(time=datetime.strptime(timestamp, "%Y%m%d%H"), y=data.y[::-1]).std(dim='number')

    else:   # model level variables
        data = xr.open_dataset(PATH_TO_DATA+f"/ERA5_EDA_ML_{DATE}.nc")
        VAR_NAME, plevel = VARIABLE[:-3], VARIABLE[-3:]
        hlevel = {"950": 96.0, "900": 106.0, "700": 119.0, "500": 123.0}

        # Variables that need scaling
        if VARIABLE in scale_factors_era5: data[VAR_NAME] *= scale_factors_era5[VARIABLE]

        # Compute statistics
        ensemble_mean = data[VAR_NAME].sel(time=datetime.strptime(timestamp, "%Y%m%d%H"), hybrid=hlevel[plevel], y=data.y[::-1]).mean(dim='number')
        ensemble_std = data[VAR_NAME].sel(time=datetime.strptime(timestamp, "%Y%m%d%H"), hybrid=hlevel[plevel], y=data.y[::-1]).std(dim='number')


    if interpolation_method is not None:
        # interpolate to finer grid
        new_y = np.linspace(ensemble_std.y.values[0], ensemble_std.y.values[-1], len(carra_dims[0]))
        new_x = np.linspace(ensemble_std.x.values[0], ensemble_std.x.values[-1], len(carra_dims[1]))
        ensemble_mean = ensemble_mean.interp(y=new_y, x=new_x, method=interpolation_method)
        ensemble_std = ensemble_std.interp(y=new_y, x=new_x, method=interpolation_method)

        # create new dataarrays from numpy arrays to remove metadata inconsistent with carra2 format
        coord_dict = {'Y': carra_dims[0], 'X': carra_dims[1]}
        ensemble_std = xr.DataArray(ensemble_std, coords=coord_dict, dims=['Y', 'X'])
        ensemble_mean = xr.DataArray(ensemble_mean, coords=coord_dict, dims=['Y', 'X'])

        # Rechunk the DataArray to match the dimension of the grid
        ensemble_std = ensemble_std.chunk({'Y': len(ensemble_std.Y), 'X': len(ensemble_std.X)})
        ensemble_mean = ensemble_mean.chunk({'Y': len(ensemble_mean.Y), 'X': len(ensemble_mean.X)})

    else:
        # create new coordinate variables for the low resolution ERA5 data
        coord_dict = {'Y_low': ensemble_std.y.values, 'X_low': ensemble_std.x.values}
        ensemble_std = xr.DataArray(ensemble_std, coords=coord_dict, dims=['Y_low', 'X_low'])
        ensemble_mean = xr.DataArray(ensemble_mean, coords=coord_dict, dims=['Y_low', 'X_low'])

        # Rechunk the DataArray to match the dimension of the grid
        ensemble_std = ensemble_std.chunk({'Y_low': len(ensemble_std.Y_low), 'X_low': len(ensemble_std.X_low)})
        ensemble_mean = ensemble_mean.chunk({'Y_low': len(ensemble_mean.Y_low), 'X_low': len(ensemble_mean.X_low)})


    # Create new Dataset containing processed data
    ds_processed = xr.Dataset({VARIABLE+"_era5_mean": ensemble_mean,
                               VARIABLE+"_era5_std": ensemble_std})

    # expand dimension (add date, to be able to append data to exisitng zarr dataset)
    ds_processed = ds_processed.expand_dims({'time': [timestamp]})

    print(f"\nERA5 variable was successfully processed: {VARIABLE}, for {timestamp}\n", flush=True)


    # plot fields
    if plot_png:
        ens_mean = ds_processed[VARIABLE+"_era5_mean"]
        ens_std = ds_processed[VARIABLE+"_era5_std"]


        # Define plotting levels if available
        if VARIABLE in variable_ranges:
            min_level, max_level = variable_ranges[VARIABLE]
            levels = np.arange(np.floor(min_level), np.ceil(max_level) + 2, 2)
        else:
            min_level = float(np.floor(ens_mean.values.min()))
            max_level = float(np.ceil(ens_mean.values.max()))
            levels = np.linspace(min_level, max_level, num=10)



        vmax = 3
        if VARIABLE == "sp": vmax = 0.1
        
        # --------------------------- Save Plots --------------------------- #
        plot_sd(
            "Standard Deviation", 
            ens_std,
            "SD_carra2.png",
            # f"{OUTDIR}/PLOTS/SD_{VARIABLE}_{YY}{MM}{DDN}_carra2.png",
            f"{YY}{MM}{DDN}",
            vmin=0,
            vmax=vmax
            #vmax=np.ceil(ens_std.values.max()),
        )

        plot_ens_mean(
            "Ensemble Mean", 
            ens_mean,
            "EnsMEAN_carra2.png",
            # f"{OUTDIR}/PLOTS/EnsMEAN_{VARIABLE}_{YY}{MM}{DDN}_carra2.png",
            f"{YY}{MM}{DDN}",
            #vmin=np.floor(ens_mean.values.min()),
            #vmax=np.ceil(ens_mean.values.max()),
            levels
        )


        # ----------------------------------------------------------------- #


        print(f"png was generated for: {VARIABLE}, for {timestamp}\n", flush=True)

    return ds_processed





# -------- Plotting Functions -------- #

def plot_ens_mean(stat_name, stat_data, output_file, DATE, levels):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    stat_data.plot(
        ax=ax, levels=levels, cmap='jet', extend='both',
        add_colorbar=True, cbar_kwargs={'label': '', 'shrink': 0.8}
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{stat_name} - Valid on {DATE}")
    plt.savefig(output_file, bbox_inches='tight', facecolor='white', dpi=200)
    plt.close()

def plot_sd(stat_name, stat_data, output_file, DATE, vmin, vmax):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    stat_data.plot(
        ax=ax, vmin=vmin, vmax=vmax, cmap='jet',
        cbar_kwargs={'label': '', 'shrink': 0.8}
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{stat_name} - Valid on {DATE}")
    plt.savefig(output_file, bbox_inches='tight', facecolor='white', dpi=200)
    plt.close()

