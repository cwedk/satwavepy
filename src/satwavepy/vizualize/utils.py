
import os, sys
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.font_manager as font_manager
import numpy as np
import math
import subprocess
import tempfile
import shutil
import os
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

import matplotlib
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.patches import PathPatch
import matplotlib.font_manager as font_manager

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import Gridliner
import cartopy
import cartopy.crs as ccrs
import colorcet as cc



def truncate_colormap(cmap, minval, maxval, n):
    """
    Creates discrete colorbar for colormap plots

    Args:
        cmap: colormap
        minval: minimum value
        maxval: maximum value
        n: number of colors
    Returns:
        new_cmap: new colormap

    """
    new_cmap = plt.cm.get_cmap(cmap)(np.linspace(minval, maxval, n))
    return ListedColormap(new_cmap)


def get_fig_size(fig_width_cm, fig_height_cm=None):
    """
    Convert dimensions in centimeters to inches.
    If no height is given, it is computed using the golden ratio.

    Args:
        fig_width_cm: figure width in cm
        fig_height_cm: figure height in cm (optional)
    Returns:
        tuple: figure size in inches
    """
    if not fig_height_cm:
        golden_ratio = (1 + math.sqrt(5)) / 2
        fig_height_cm = fig_width_cm / golden_ratio

    size_cm = (fig_width_cm, fig_height_cm)
    return tuple(map(lambda x: x / 2.54, size_cm))  # in cm : x /2.54


def label_size():
    """Size of axis labels"""
    return 7


def font_size():
    """Size of all texts shown in plots"""
    return 7


def legend_font_size():
    """Size of legend font"""
    return 7


def ticks_size():
    """Size of axes' ticks"""
    return 6


def axis_lw():
    """Line width of the axes"""
    return 0.4


def plot_lw():
    """Line width of the plotted curves"""
    return 1


def figure_setup():
    """
    Set all the sizes to the correct values and use
    tex fonts for all texts.
    """
    # Set the font path and font properties manually
    font_path = "/p/i1000617-phd-waveda/fonts/Helvetica.ttf"
    font_manager.fontManager.addfont(font_path)
    font_prop = font_manager.FontProperties(fname=font_path)

    params = {
        "text.usetex": False,
        "figure.dpi": 300,
        "font.size": font_size(),
        "font.sans-serif": [font_prop.get_name()],  # Use the Helvetica font
        "axes.labelsize": label_size(),
        "legend.fontsize": legend_font_size(),
        "axes.titlesize": font_size(),
        "axes.linewidth": axis_lw(),
        "xtick.labelsize": ticks_size(),
        "ytick.labelsize": ticks_size(),
        "font.family": font_prop.get_name(),
        "legend.frameon": False,
    }  # Set the font family to Helvetica
    plt.rcParams.update(params)


def adjust_legend_position(ax, legend):
    """
    Adjust the legend position to avoid overlapping with the data.
    
    Args:
        ax (matplotlib.axes.Axes): axis
        legend (matplotlib.legend.Legend): legend
    Returns:
        None
    """
    ax_position = ax.get_position()
    legend_position = legend.get_window_extent().transformed(ax.transData.inverted())
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Check if the legend overlaps with the data
    if legend_position.x0 < x_max and legend_position.y0 < y_max:
        # Adjust the legend position to the right and above the plot area
        new_x = x_max + (legend_position.x1 - legend_position.x0) * 1.05
        new_y = y_max + (legend_position.y1 - legend_position.y0) * 1.05
        legend.set_bbox_to_anchor((new_x, new_y), transform=ax.transData)


def save_fig(fig, file_name, fmt=None, dpi=300, tight=True):
    """
    Save a Matplotlib figure as EPS/PNG/PDF to the given path and trim it.
    
    Args:
        fig (matplotlib.figure.Figure): figure
        file_name (str): path to save the figure
        fmt (str): format of the figure (eps, png, pdf)
        dpi (int): resolution of the figure
        tight (bool): trim the figure
    Returns:
        custom_lat_formatter (function): custom latitude formatter
    """

    if not fmt:
        fmt = file_name.strip().split(".")[-1]

    if fmt not in ["eps", "png", "pdf"]:
        raise ValueError("unsupported format: %s" % (fmt,))

    extension = ".%s" % (fmt,)
    if not file_name.endswith(extension):
        file_name += extension

    file_name = os.path.abspath(file_name)
    with tempfile.NamedTemporaryFile() as tmp_file:
        tmp_name = tmp_file.name + extension

    # save figure
    if tight:
        fig.savefig(tmp_name, dpi=dpi, bbox_inches="tight")

        # if not trim, move or copy the temporary file to the specified path
        shutil.move(tmp_name, file_name)
    else:
        fig.savefig(tmp_name, dpi=dpi)

        # if not trim, move or copy the temporary file to the specified path
        shutil.move(tmp_name, file_name)

    # trim it
    # if fmt == 'eps':
    #     subprocess.call('epstool --bbox --copy %s %s' %
    #                     (tmp_name, file_name), shell=True)
    # elif fmt == 'png':
    #     subprocess.call('convert %s -trim %s' %
    #                     (tmp_name, file_name), shell=True)
    # elif fmt == 'pdf':
    #     subprocess.call('pdfcrop %s %s' % (tmp_name, file_name), shell=True)

def setup_custom_lat_formatter():
    def custom_lat_formatter(x, pos=None):
        """
        formats the latitudes in the map
        """
        if x % 2 != 1:
            if x < 0:
                return rf"-{abs(x)}$^\circ$"
            elif x > 0:
                return rf"{x}$^\circ$"
            else:
                return rf"{x}$^\circ$"
        else:
            return ""

    return custom_lat_formatter


def create_projection():
    """
    Create orthographic projection centered on the domain.

    Args:
        None
    Returns:
        noProj (cartopy.crs.Projection): projection
        myProj (cartopy.crs.Projection): projection
    """
    noProj = ccrs.PlateCarree(central_longitude=4)
    myProj = ccrs.Orthographic(central_longitude=4, central_latitude=56)  # -2.5, 70
    myProj._threshold = myProj._threshold / 40.0  # for higher precision plot
    return noProj, myProj



def create_figure_and_axes(projection, figuresize):
    """
    Creates figure and axes.

    Args:
        projection (cartopy.crs.Projection): projection
        figuresize (tuple): figure size
    Returns:
        fig (matplotlib.figure.Figure): figure
        ax (matplotlib.axes.Axes): axis
    """
    fig, ax = plt.subplots(figsize=figuresize, subplot_kw={"projection": projection})
    return fig, ax


def plot_zebra_border(ax, noProj, myProj):
    """
    Add zebra border to spatial map plot. Note the offset of 4 longitude used
    to create orthographic projection.

    Args:
        ax (matplotlib.axes.Axes): axis
        noProj (cartopy.crs.Projection): projection
        myProj (cartopy.crs.Projection): projection
    Returns:
        polygon1s (matplotlib.path.Path): polygon
    """

    offset = 4  # see central longitude in projection
    x_coords_orig = np.array([-5.0 - offset, 13 - offset, 13 - offset, -5.0 - offset])
    y_coords_orig = np.array([50, 50, 62, 62])

    # Interpolate the coordinates to create a higher resolution polygon
    num_points = 100  # Increase this value for a smoother boundary
    t = np.linspace(0, 1, num_points)
    x_coords = np.interp(t, np.linspace(0, 1, len(x_coords_orig)), x_coords_orig)
    y_coords = np.interp(t, np.linspace(0, 1, len(y_coords_orig)), y_coords_orig)

    # Close the polygon by repeating the first point
    x_coords = np.append(x_coords, x_coords[0])
    y_coords = np.append(y_coords, y_coords[0])

    [ax_hdl] = ax.plot(
        x_coords, y_coords, color="black", linewidth=0.5, transform=noProj
    )
    tx_path = ax_hdl._get_transformed_path()
    path_in_data_coords, _ = tx_path.get_transformed_path_and_affine()
    polygon1s = mpath.Path(path_in_data_coords.vertices, closed=True)
    ax.set_boundary(polygon1s)  # masks-out unwanted part of the plot

    for artist in ax.get_children():
        if isinstance(artist, matplotlib.image.AxesImage):
            artist.set_clip_path(
                PathPatch(polygon1s, transform=myProj._as_mpl_transform(ax))
            )

    return polygon1s


def add_model_limits(ax, noProj):
    """
    Add model domain to spatial map plot. Note the offset of 2.5  longitude used
    to create orthographic projection

    Args:
        ax (matplotlib.axes.Axes): axis
        noProj (cartopy.crs.Projection): projection
    Returns:
        ax (matplotlib.axes.Axes): axis

    """
    from shapely.geometry import Polygon, LineString
    from shapely.geometry.polygon import orient
    from shapely.geometry import Polygon

    # Corners
    offset = -4
    lon1 = -3.25 + offset  # -15+2.5
    lat1 = 51.25  # 43
    lon2 = 10.25 + offset  # 13+2.5
    lat2 = 61.75  # 64

    lower_left = [lon1, lat1]
    upper_left = [lon1, lat2]
    upper_right = [lon2, lat2]
    lower_right = [lon2, lat1]  # [lon2, lat1+7]
    lower_central = [lon2 - 7, lat1]  # [lon2-13, lat1]

    # Define the original coordinates
    x_coords_orig = np.array(
        [lower_left[0], upper_left[0], upper_right[0], lower_right[0], lower_central[0]]
    )
    y_coords_orig = np.array(
        [lower_left[1], upper_left[1], upper_right[1], lower_right[1], lower_central[1]]
    )

    # Interpolate the coordinates to create a higher resolution polygon
    num_points = 1000  # Increase this value for a smoother boundary
    t = np.linspace(0, 1, num_points)
    x_coords = np.interp(t, np.linspace(0, 1, len(x_coords_orig)), x_coords_orig)
    y_coords = np.interp(t, np.linspace(0, 1, len(y_coords_orig)), y_coords_orig)

    # Close the polygon by repeating the first point
    x_coords = np.append(x_coords, x_coords_orig[0])
    y_coords = np.append(y_coords, y_coords_orig[0])

    x_coords = np.append(x_coords, x_coords_orig[1])
    y_coords = np.append(y_coords, y_coords_orig[1])

    # Create the polygon from the interpolated coordinates
    polygon_coords = list(zip(x_coords, y_coords))
    polygon = Polygon(polygon_coords)

    # Add the polygon to the plot
    model_limits = ax.add_geometries(
        [polygon], noProj, facecolor="none", edgecolor="red", label="Model limits"
    )

    return ax


def add_features(ax):
    """
    Add features to axis

    Args:
    ax (matplotlib.axes._subplots.AxesSubplot): Matplotlib axis to plot on.

    Returns:
    matplotlib.axes._subplots.AxesSubplot: Matplotlib axis with features added.
    """
    cmap = cc.cm.glasbey_bw
    cmap_warm = cc.cm.glasbey_warm
    color_beige = cmap.colors[60]
    font_prop = font_manager.FontProperties(family="Helvetica")

    LAND = cartopy.feature.NaturalEarthFeature(
        "physical", "land", scale="10m", zorder=10, edgecolor=color_beige
    )
    BORDERS = cartopy.feature.NaturalEarthFeature(
        "cultural",
        "admin_0_boundary_lines_land",
        "10m",
        edgecolor="grey",
        facecolor="none",
    )
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, zorder=10)
    gl.yformatter = mticker.FuncFormatter(setup_custom_lat_formatter())
    gl.xformatter = mticker.FuncFormatter(setup_custom_lat_formatter())
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}
    ax.add_feature(LAND, zorder=3, color=color_beige, edgecolor=color_beige)
    ax.add_feature(BORDERS, zorder=4, lw=0.3)
    gl.xlocator = mticker.FixedLocator(np.arange(-10, 20, 1))
    gl.ylocator = mticker.FixedLocator(np.arange(48, 66, 1))
    ax.coastlines(
        resolution="10m",
        linewidth=0.2,
        zorder=5,
        edgecolor=color_beige,
        facecolor="none",
    )

    return ax, gl


def plot_satellite_tracks(ax, xtrack_names, noProj, settings=0):
    pst = ax.scatter(
        xtrack_names["x"].values + 2.5,
        xtrack_names["y"].values,
        s=0.5,
        lw=0.1,
        c="white",
        marker="o",
        transform=noProj,
        zorder=10,
        label="Satellite tracks",
    )
    return pst


def plot_swan_grid(ax, xpc, ypc, xlenc, ylenc, mxc, myc):
    """
    Plots a grid from .swn grid params

    Args:
        ax (matplotlib.axes._subplots.AxesSubplot): Matplotlib axis to plot on.
        xpc (float): x origin of the grid
        ypc (float): y origin of the grid
        xlenc (float): grid dimensions in degrees
        ylenc (float): grid dimensions in degrees
        mxc (int): grid resolution
        myc (int): grid resolution

    Returns:
    matplotlib.axes._subplots.AxesSubplot: Matplotlib axis with grid added.
    """
    # Grid parameters extracted from the CGRID command
    xpc, ypc = -3.0, 51.5  # Origin of the grid
    xlenc, ylenc = 13.0, 10.0  # Grid dimensions in degrees
    mxc, myc = 26, 20  # Grid resolution

    # Calculate grid points
    x = np.linspace(xpc, xpc + xlenc, mxc + 1)
    y = np.linspace(ypc, ypc + ylenc, myc + 1)
    X, Y = np.meshgrid(x, y)

    print(X.shape)
    print(Y.shape)

    # Plot grid lines
    for i in range(
        myc + 1
    ):  # mxc is the number of divisions, so the last index is mxc - 1
        ax.plot(X[i, :], Y[i, :], "k-", transform=ccrs.Geodetic(), linewidth=0.1)
    for j in range(
        mxc + 1
    ):  # myc is the number of divisions, so the last index is myc - 1
        ax.plot(X[:, j], Y[:, j], "k-", transform=ccrs.Geodetic(), linewidth=0.1)

    return ax

