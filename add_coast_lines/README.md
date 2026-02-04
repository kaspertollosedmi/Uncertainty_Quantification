# CARRA2 Coastline Plotting Tool

Add coastlines and geographic features to CARRA2 uncertainty field plots from netCDF files.

## Features

- Plots netCDF data on North Polar Stereographic projection
- Adds coastlines and country borders
- Optional latitude/longitude gridlines with labels
- Optional colorbar
- Automatic detection and removal of NaN padding

## Quick Start

### 1. Create the conda environment

```bash
conda env create -f environment.yml
```

This creates an environment called `coastline_plotting` with all required dependencies.

### 2. Run with the batch script

Edit `run_coastlines.sh` to configure:
- `DATETIMES`: List of datetime strings to process
- `INPUT_DIR` / `OUTPUT_DIR`: Input/output directories
- `SHOW_COLORBAR`: Set to 1 to show colorbar, 0 to hide
- `SHOW_GRIDLINES`: Set to 1 to show lat/lon gridlines, 0 to hide
- `VMIN` / `VMAX`: Color scale limits

Then run:
```bash
chmod +x run_coastlines.sh
./run_coastlines.sh
```

### 3. Run for a single datetime

```bash
conda activate coastline_plotting
python add_coastlines.py 2019050100 --colorbar --gridlines
```

## Command-Line Usage

```
usage: add_coastlines.py [-h] [--input-dir INPUT_DIR] [--output-dir OUTPUT_DIR]
                         [--colorbar] [--gridlines] [--vmin VMIN] [--vmax VMAX]
                         datetime

Add coastlines to CARRA2 uncertainty field plots

positional arguments:
  datetime              DateTime string in format YYYYMMDDHH (e.g., 2019050100)

optional arguments:
  -h, --help            show this help message and exit
  --input-dir, -i       Input directory containing netCDF files
                        (default: ../sample_data/input)
  --output-dir, -o      Output directory for PNG files
                        (default: ../sample_data/output)
  --colorbar, -c        Add colorbar to the plot
  --gridlines, -g       Add lat/lon gridlines with labels
  --vmin VMIN           Minimum value for color scale (default: 0.0)
  --vmax VMAX           Maximum value for color scale (default: 3.0)
```

## Dependencies

- Python >= 3.9
- numpy
- xarray
- netcdf4
- matplotlib
- cartopy
- pyproj

## Input Data

The script expects netCDF files named `UQ_<datetime>.nc` with 2D data arrays. The data is assumed to be on the CARRA2 West domain grid (North Polar Stereographic projection).

Grid parameters used:
- Domain center: 45°W, 84°N
- Projection reference: 30°W, 90°N (North Pole)
- Grid resolution: ~2.6 km (adjusted for 2766x2766 grid after padding removal)

## Output

PNG files named `UQ_<datetime>.png` are saved to the output directory.
