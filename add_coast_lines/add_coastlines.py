# -*- coding: utf-8 -*-
"""
Add coastlines to netCDF field plots using cartopy.

The netCDF files from the diffusion model lack coordinate metadata,
so we need to define the CARRA2 grid projection parameters manually.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# CARRA2 West domain grid parameters (North Polar Stereographic)
# Original model config: NLON=2880, NLAT=2880, LONC=-45, LATC=84, LON0=-30, LAT0=90, GSIZE=2500
# Adjusted for diffusion model output with 57-pixel padding removed on each side
CARRA2_PARAMS = {
    # Projection parameters for North Polar Stereographic for CARRA2
    'central_longitude': -30.0,   # LON0 - projection reference longitude
    'true_scale_latitude': 90.0,  # LAT0 - North Pole
    'domain_center_lon': -45.0,   # LONC
    'domain_center_lat': 84.0,    # LATC
    'grid_resolution': 2500.0,    # GSIZE - meters (2.5 km)
    'nx': 2880,                   # NLON
    'ny': 2880,                   # NLAT
}

CARRA2_WITH_PADDING_PARAMS = {
    # Projection parameters for North Polar Stereographic for CARRA2
    'central_longitude': -30.0,   # LON0 - projection reference longitude
    'true_scale_latitude': 90.0,  # LAT0 - North Pole
    'domain_center_lon': -45.0,   # LONC
    'domain_center_lat': 84.0,    # LATC
    # changed coordinate parameters due to padding removal:
    'grid_resolution': 2603.07,   # GSIZE - meters (2.5 km) - changed due to grid number adjustment
    'nx': 2766,                   # NLON after padding removal (2880 - 2*57)
    'ny': 2766,                   # NLAT after padding removal
}


def calculate_grid_extent(params):
    """Calculate grid origin from domain center coordinates."""
    import pyproj

    # Create the projection
    proj = pyproj.Proj(proj='stere', lat_0=90, lon_0=params['central_longitude'],
                       lat_ts=params['true_scale_latitude'])

    # Convert domain center to projection coordinates
    x_center, y_center = proj(params['domain_center_lon'], params['domain_center_lat'])

    # Calculate grid origin (lower-left corner)
    dx = params['grid_resolution']
    nx = params['nx']
    ny = params['ny']

    x_origin = x_center - (nx / 2) * dx
    y_origin = y_center - (ny / 2) * dx

    return x_origin, y_origin, x_center, y_center


def detect_padding(data, threshold=0.0):
    """
    Detect padding around the edges of the data.

    Looks for rows/columns that are all NaN or have very low variance.

    Returns: dict with padding info and the indices of actual data.
    """
    ny, nx = data.shape

    # Check for NaN values
    nan_count = np.isnan(data).sum()
    nan_fraction = nan_count / data.size

    print("\n=== Padding Detection ===")
    print(f"Data shape: {ny} x {nx}")
    print(f"NaN count: {nan_count} ({nan_fraction*100:.1f}% of data)")

    # Check if padding is NaN-based
    if nan_count > 0:
        print("Padding appears to be NaN values")

        # Count valid (non-NaN) values per row and column
        valid_per_row = np.sum(~np.isnan(data), axis=1)
        valid_per_col = np.sum(~np.isnan(data), axis=0)

        # Find first/last rows/cols with any valid data
        row_has_data = valid_per_row > 0
        col_has_data = valid_per_col > 0

        first_row = np.argmax(row_has_data)
        last_row = ny - 1 - np.argmax(row_has_data[::-1])
        first_col = np.argmax(col_has_data)
        last_col = nx - 1 - np.argmax(col_has_data[::-1])
    else:
        # Fall back to variance-based detection
        row_std = np.nanstd(data, axis=1)
        col_std = np.nanstd(data, axis=0)

        row_has_data = row_std > threshold
        col_has_data = col_std > threshold

        first_row = np.argmax(row_has_data) if row_has_data.any() else 0
        last_row = ny - 1 - np.argmax(row_has_data[::-1]) if row_has_data.any() else ny - 1
        first_col = np.argmax(col_has_data) if col_has_data.any() else 0
        last_col = nx - 1 - np.argmax(col_has_data[::-1]) if col_has_data.any() else nx - 1

    padding = {
        'top': first_row,
        'bottom': ny - 1 - last_row,
        'left': first_col,
        'right': nx - 1 - last_col,
        'data_slice': (slice(first_row, last_row + 1), slice(first_col, last_col + 1)),
        'data_shape': (last_row - first_row + 1, last_col - first_col + 1),
    }

    print(f"\nPadding detected (in pixels):")
    print(f"  Top: {padding['top']}, Bottom: {padding['bottom']}")
    print(f"  Left: {padding['left']}, Right: {padding['right']}")
    print(f"  Actual data region: rows [{first_row}:{last_row+1}], cols [{first_col}:{last_col+1}]")
    print(f"  Actual data shape: {padding['data_shape']}")

    # Check if padding is symmetric
    if padding['top'] == padding['bottom'] and padding['left'] == padding['right']:
        print(f"  Padding is symmetric!")
    else:
        print(f"  Warning: Padding is NOT symmetric")

    # Get stats on actual data region
    actual_data = data[padding['data_slice']]
    print(f"\nActual data stats (excluding padding):")
    print(f"  Min: {np.nanmin(actual_data):.4f}, Max: {np.nanmax(actual_data):.4f}")
    print(f"  Mean: {np.nanmean(actual_data):.4f}, Std: {np.nanstd(actual_data):.4f}")

    return padding


def read_netcdf(nc_path):
    """Read netCDF file and return data array."""
    ds = xr.open_dataset(nc_path)
    print("=== Dataset Info ===")
    print(ds)
    print("\n=== Attributes ===")
    print(ds.attrs)
    print("\n=== Data Variables ===")
    for var in ds.data_vars:
        da = ds[var]
        print(f"  {var}: shape={da.shape}, min={float(da.min()):.4f}, max={float(da.max()):.4f}")
    return ds


def plot_with_coastlines(data, output_file, cmap='jet', vmin=None, vmax=None,
                         padding=None, colorbar=False, gridlines=False):
    """
    Plot data with coastlines using cartopy.

    This requires knowing the grid extent in the projection coordinates.
    If padding is provided, crop the data and adjust extent accordingly.

    Parameters
    ----------
    data : array-like
        2D data array to plot
    output_file : str
        Path to save the output figure
    cmap : str
        Colormap name (default: 'jet')
    vmin, vmax : float
        Color scale limits
    padding : dict
        Padding info from detect_padding() to crop NaN borders
    colorbar : bool
        Whether to add a colorbar (default: False)
    gridlines : bool
        Whether to add latitude/longitude gridlines with labels (default: False)
    """
    projection = ccrs.NorthPolarStereo(
        central_longitude=CARRA2_PARAMS['central_longitude']
    )

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': projection})

    # Crop data if padding info provided
    if padding is not None and padding['top'] > 0:
        data = data[padding['data_slice']]
        print(f"Cropped data shape: {data.shape}")

    # Calculate extent from domain center (CARRA2_PARAMS now has cropped dimensions)
    x0, y0, x_center, y_center = calculate_grid_extent(CARRA2_PARAMS)
    if padding is None:
        dx = CARRA2_PARAMS['grid_resolution']
        nx = CARRA2_PARAMS['nx']
        ny = CARRA2_PARAMS['ny']
    else:
        dx = CARRA2_WITH_PADDING_PARAMS['grid_resolution']
        nx = CARRA2_WITH_PADDING_PARAMS['nx']
        ny = CARRA2_WITH_PADDING_PARAMS['ny']

    x1 = x0 + nx * dx
    y1 = y0 + ny * dx

    print(f"Grid extent: x=[{x0:.0f}, {x1:.0f}], y=[{y0:.0f}, {y1:.0f}]")
    print(f"Domain center: ({x_center:.0f}, {y_center:.0f})")

    # extent = [left, right, bottom, top]
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower',
                   extent=[x0, x1, y0, y1], transform=projection)

    # Add coastlines and features
    ax.coastlines(resolution='50m', color='black', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)

    # Add gridlines with labels if requested
    if gridlines:
        import matplotlib.ticker as mticker
        gl = ax.gridlines(draw_labels=True, linewidth=1., color='gray',
                          alpha=0.7, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 8, 'color': 'black'}
        gl.ylabel_style = {'size': 8, 'color': 'black'}
        # Set longitude lines every 10 degrees (doubled density)
        gl.xlocator = mticker.FixedLocator(range(-180, 181, 20))
        gl.ylocator = mticker.FixedLocator(range(50, 91, 10))

    if colorbar:
        plt.colorbar(im, ax=ax, shrink=0.7, label='')

    plt.savefig(output_file, bbox_inches='tight', facecolor='white', dpi=150)
    plt.close()
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Add coastlines to CARRA2 uncertainty field plots"
    )
    parser.add_argument(
        "datetime",
        help="DateTime string in format YYYYMMDDHH (e.g., 2019050100)"
    )
    parser.add_argument(
        "--input-dir", "-i",
        default="../sample_data/input",
        help="Input directory containing netCDF files (default: ../sample_data/input)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="../sample_data/output",
        help="Output directory for PNG files (default: ../sample_data/output)"
    )
    parser.add_argument(
        "--colorbar", "-c",
        action="store_true",
        help="Add colorbar to the plot"
    )
    parser.add_argument(
        "--gridlines", "-g",
        action="store_true",
        help="Add lat/lon gridlines with labels"
    )
    parser.add_argument(
        "--vmin",
        type=float, default=0.0,
        help="Minimum value for color scale (default: 0.0)"
    )
    parser.add_argument(
        "--vmax",
        type=float, default=3.0,
        help="Maximum value for color scale (default: 3.0)"
    )

    args = parser.parse_args()

    # Build file paths
    nc_file = f"{args.input_dir}/UQ_{args.datetime}.nc"
    out_file = f"{args.output_dir}/UQ_{args.datetime}.png"

    # Read and inspect data
    ds = read_netcdf(nc_file)

    # Get the data variable (assuming single variable or 'UQ')
    var_names = list(ds.data_vars)
    var_name = var_names[0]
    data = ds[var_name].values

    print(f"\nPlotting variable: {var_name}")
    print(f"Shape: {data.shape}")

    # Detect padding
    padding = detect_padding(data)

    # Plot with coastlines
    plot_with_coastlines(
        data, out_file,
        vmin=args.vmin, vmax=args.vmax,
        padding=padding,
        colorbar=args.colorbar,
        gridlines=args.gridlines
    )

    print("\nDone!")
