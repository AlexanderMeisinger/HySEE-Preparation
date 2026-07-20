# pv- and wind potential analysis
# Base: Timon Geiss, OTH Regensburg
# Adapted: Alexander Meisinger, OTH Regensburg

# Data Class is not resolved immediately when the code is parsed, Python stores them in a delayed form and evaluates them later when needed
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# NetCDF-4 data type is intern based on HDF5
import h5py
import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch

from shapely.geometry import shape


# PATHS AND FILE NAMES
show_duration_curves = False

# Adjust the pattern if needed to match the actual result network file name in your project.
country = "Romania" # Bulgaria, Romania
project_dir = Path(f"/mnt/e/HySEE/{country}/pypsa-earth")
figure_dir = Path(f"/mnt/e/HySEE/HySEE-Preparation/workflow/figures/{country}")
table_dir = Path(f"/mnt/e/HySEE/HySEE-Preparation/workflow/tables/{country}")

scenario = "1h-sec"
RESULT_NETWORK_FILE_PATTERN = "elec_s_10_ec_lcopt_Co2L0.45-1h_1h_2030_0.07_AB_0export.nc"

VOLTAGE_COLORS = {
    110000: "#4c78a8",
    220000: "#f58518",
    400000: "#54a24b",
}


# DATACLASS DEFINITIONS

@dataclass(frozen=True)
class ProjectPaths:
    country_name: str
    base_network_dir: Path
    country_shape_file: Path
    result_network_file: Path
    solar_profile_file: Path
    wind_profile_file: Path
    profile_busmap_file: Path
    profile_regions_file: Path
    figure_dir: Path
    base_network_output_file: Path
    result_network_output_file: Path
    result_network_labeled_output_file: Path
    solar_potential_output_file: Path
    wind_potential_output_file: Path
    solar_potential_median_output_file: Path
    wind_potential_median_output_file: Path


def load_geojson(path: Path) -> dict | None:
    """Load a GeoJSON file from disk.

    Args:
        path: Path to the GeoJSON file.

    Returns:
        The parsed GeoJSON object or None when the file is empty.
    """
    # Read the full file first so we can detect empty files.
    text = path.read_text(encoding="utf-8")

    # Return None instead of crashing when the file only contains whitespace.
    if not text.strip():
        return None

    # Parse the JSON text into a normal Python dictionary.
    return json.loads(text)


def iter_line_segments(coordinates: list) -> Iterable[tuple[list[float], list[float]]]:
    """Yield x/y coordinate sequences for line geometries.

    Args:
        coordinates: Coordinate block from a LineString or MultiLineString geometry.

    Returns:
        An iterator of x/y coordinate tuples.
    """
    # Stop early when there is no geometry data at all.
    if not coordinates:
        return

    # Look at the first item to decide whether we already have one line
    # or a nested structure with multiple lines.
    first_item = coordinates[0]
    is_single_line = (
        isinstance(first_item, list)
        and len(first_item) >= 2
        and isinstance(first_item[0], (int, float))
        and isinstance(first_item[1], (int, float))
    )

    if is_single_line:
        x_values: list[float] = []
        y_values: list[float] = []
        for point in coordinates:
            x_values.append(point[0])
            y_values.append(point[1])
        yield x_values, y_values
        return

    # For nested coordinate lists we simply walk one level deeper.
    for nested_coordinates in coordinates:
        yield from iter_line_segments(nested_coordinates)


def iter_polygon_rings(coordinates: list) -> Iterable[tuple[list[float], list[float]]]:
    """Yield x/y coordinate sequences for polygon boundaries.

    Args:
        coordinates: Coordinate block from a Polygon or MultiPolygon geometry.

    Returns:
        An iterator of x/y coordinate tuples.
    """
    # Stop early when the geometry has no coordinates.
    if not coordinates:
        return

    # Decide whether the current level already contains polygon rings.
    first_item = coordinates[0]
    is_polygon_level = (
        isinstance(first_item, list)
        and first_item
        and isinstance(first_item[0], list)
        and len(first_item[0]) >= 2
        and isinstance(first_item[0][0], (int, float))
        and isinstance(first_item[0][1], (int, float))
    )

    if is_polygon_level:
        for ring in coordinates:
            x_values: list[float] = []
            y_values: list[float] = []
            for point in ring:
                x_values.append(point[0])
                y_values.append(point[1])
            yield x_values, y_values
        return

    # For MultiPolygon-style nesting we move one level deeper.
    for nested_coordinates in coordinates:
        yield from iter_polygon_rings(nested_coordinates)


def ensure_output_dir(path: Path) -> None:
    """Create an output directory when it does not exist.

    Args:
        path: Directory that should exist before writing output files.

    Returns:
        None.
    """
    # Create the full folder path and do nothing when it already exists.
    path.mkdir(parents=True, exist_ok=True)


def load_hdf5_strings(file_handle: h5py.File, dataset_name: str) -> list[str]:
    """Read a string dataset from an HDF5 file.

    Args:
        file_handle: Open HDF5 file handle.
        dataset_name: Name of the dataset to read.

    Returns:
        The dataset values converted to Python strings.
    """
    # Collect converted values step by step to keep the logic readable.
    values: list[str] = []

    for value in file_handle[dataset_name][...]:
        # Most string datasets are stored as bytes and need decoding.
        if isinstance(value, bytes):
            values.append(value.decode("utf-8"))
            continue

        # Some values expose a decode method even when they are not plain bytes.
        if hasattr(value, "decode"):
            values.append(value.decode("utf-8"))
            continue

        # Fallback to a plain string conversion for everything else.
        if hasattr(value, "item"):
            values.append(str(value.item()))
        else:
            values.append(str(value))

    return values


def load_hdf5_floats(file_handle: h5py.File, dataset_name: str) -> list[float]:
    """Read a numeric dataset from an HDF5 file.

    Args:
        file_handle: Open HDF5 file handle.
        dataset_name: Name of the dataset to read.

    Returns:
        The dataset values converted to floats.
    """
    # Convert each numeric value one by one for maximum clarity.
    float_values: list[float] = []

    for value in file_handle[dataset_name][...]:
        float_values.append(float(value))

    return float_values

def load_renewable_profile_data(profile_file: Path) -> dict:
    """Read time series and potential data from a renewable profile file.

    Args:
        profile_file: Path to the NetCDF profile file.

    Returns:
        A dictionary with buses, timestamps, profiles, capacities, and potential grid data.
    """
    # Open the profile file only for the time we actually need to read it.
    with h5py.File(profile_file, "r") as profile_data:
        # Read the raw hour offsets and convert them into a proper time axis.
        time_values = np.array(profile_data["time"][...], dtype=float)
        time_units = profile_data["time"].attrs["units"]
        if isinstance(time_units, bytes):
            time_units = time_units.decode("utf-8")

        time_start = time_units.split("since", maxsplit=1)[1].strip()
        time_index = pd.to_datetime(time_start) + pd.to_timedelta(time_values, unit="h")

        # Read the remaining arrays exactly once and store them in a plain dictionary.
        profile_dict: dict = {}
        profile_dict["bus"] = load_hdf5_strings(profile_data, "bus")
        profile_dict["time"] = time_index
        profile_dict["profile"] = np.array(profile_data["profile"][...], dtype=float)
        profile_dict["p_nom_max"] = np.array(profile_data["p_nom_max"][...], dtype=float)
        profile_dict["x"] = np.array(profile_data["x"][...], dtype=float)
        profile_dict["y"] = np.array(profile_data["y"][...], dtype=float)
        profile_dict["potential"] = np.array(profile_data["potential"][...], dtype=float)

    return profile_dict


def load_result_network_data(network_file: Path) -> dict:
    """Read the bus, line, and link datasets from one result network file.

    Args:
        network_file: Path to the solved PyPSA network file.

    Returns:
        A dictionary with the bus coordinates and the line and link columns
        needed by the plotting and CSV export functions.
    """
    # Open the network file only once and copy every required dataset into plain arrays.
    with h5py.File(network_file, "r") as network_data:
        network_dict: dict = {}

        # Read the bus table so later code can look up coordinates by bus name.
        network_dict["bus_names"] = load_hdf5_strings(network_data, "buses_i")
        network_dict["bus_x_values"] = load_hdf5_floats(network_data, "buses_x")
        network_dict["bus_y_values"] = load_hdf5_floats(network_data, "buses_y")

        # Read the main line columns that are useful for both plotting and table export.
        network_dict["line_names"] = load_hdf5_strings(network_data, "lines_i")
        network_dict["line_bus0"] = load_hdf5_strings(network_data, "lines_bus0")
        network_dict["line_bus1"] = load_hdf5_strings(network_data, "lines_bus1")
        network_dict["line_carriers"] = load_hdf5_strings(network_data, "lines_carrier")
        network_dict["line_lengths"] = load_hdf5_floats(network_data, "lines_length")
        network_dict["line_v_nom"] = load_hdf5_floats(network_data, "lines_v_nom")
        network_dict["line_s_nom"] = load_hdf5_floats(network_data, "lines_s_nom")

        # Read the main link columns in the same explicit style.
        network_dict["link_names"] = load_hdf5_strings(network_data, "links_i")
        network_dict["link_bus0"] = load_hdf5_strings(network_data, "links_bus0")
        network_dict["link_bus1"] = load_hdf5_strings(network_data, "links_bus1")
        network_dict["link_carriers"] = load_hdf5_strings(network_data, "links_carrier")
        network_dict["link_p_nom_opt"] = load_hdf5_floats(network_data, "links_p_nom_opt")

    return network_dict


def load_profile_regions(
    profile_bus_names: list[str],
    busmap_file: Path,
    profile_regions_file: Path,
) -> list[dict]:
    """Map renewable profile buses to their matching region polygons.

    Args:
        profile_bus_names: Bus names from the renewable profile file.
        busmap_file: CSV file mapping buses to region identifiers.
        profile_regions_file: GeoJSON file with onshore region geometries.

    Returns:
        Region metadata in the same order as the incoming profile buses.
        Each region dictionary contains the region name, the polygon geometry,
        and the calculated ``x`` and ``y`` center coordinates that are used to
        place the numeric labels in the PV and wind median overview plots.
    """
    def get_outer_rings(geometry: dict) -> list[list[list[float]]]:
        """Extract outer polygon rings from Polygon or MultiPolygon geometry."""
        geometry_type = geometry.get("type")
        geometry_coordinates = geometry.get("coordinates", [])
        outer_rings: list[list[list[float]]] = []

        if geometry_type == "Polygon":
            if geometry_coordinates:
                outer_rings.append(geometry_coordinates[0])
            return outer_rings

        if geometry_type == "MultiPolygon":
            for polygon_coordinates in geometry_coordinates:
                if polygon_coordinates:
                    outer_rings.append(polygon_coordinates[0])
            return outer_rings

        return outer_rings

    def calculate_ring_area(ring: list[list[float]]) -> float:
        """Calculate the signed area of one polygon ring."""
        if len(ring) < 3:
            return 0.0

        area = 0.0
        point_count = len(ring)
        for index in range(point_count):
            current_point = ring[index]
            next_point = ring[(index + 1) % point_count]
            area += current_point[0] * next_point[1]
            area -= next_point[0] * current_point[1]

        return area / 2.0

    def calculate_ring_center(ring: list[list[float]]) -> tuple[float, float]:
        """Calculate a simple polygon centroid from one outer ring."""
        if len(ring) < 3:
            if not ring:
                return 0.0, 0.0
            return float(ring[0][0]), float(ring[0][1])

        signed_area = calculate_ring_area(ring)
        if abs(signed_area) < 1e-12:
            # Fall back to a plain average when the polygon area is too small.
            x_sum = 0.0
            y_sum = 0.0
            for point in ring:
                x_sum += float(point[0])
                y_sum += float(point[1])
            return x_sum / len(ring), y_sum / len(ring)

        centroid_x = 0.0
        centroid_y = 0.0
        point_count = len(ring)
        for index in range(point_count):
            current_point = ring[index]
            next_point = ring[(index + 1) % point_count]
            cross_product = (current_point[0] * next_point[1]) - (next_point[0] * current_point[1])
            centroid_x += (current_point[0] + next_point[0]) * cross_product
            centroid_y += (current_point[1] + next_point[1]) * cross_product

        centroid_x = centroid_x / (6.0 * signed_area)
        centroid_y = centroid_y / (6.0 * signed_area)
        return float(centroid_x), float(centroid_y)

    def calculate_geometry_center(geometry: dict) -> tuple[float, float]:
        """Use the largest outer ring so labels stay centered in the visible region."""
        outer_rings = get_outer_rings(geometry)
        if not outer_rings:
            return 0.0, 0.0

        largest_ring = outer_rings[0]
        largest_area = abs(calculate_ring_area(largest_ring))
        for ring in outer_rings[1:]:
            current_area = abs(calculate_ring_area(ring))
            if current_area > largest_area:
                largest_ring = ring
                largest_area = current_area

        return calculate_ring_center(largest_ring)

    # Read the bus-to-region mapping table and keep only valid rows.
    busmap_table = pd.read_csv(busmap_file)
    busmap_table = busmap_table.dropna(subset=["0"])

    # Build a plain dictionary that maps each bus name to one region id.
    bus_to_region: dict[str, str] = {}
    for bus_value, region_value in zip(busmap_table["Unnamed: 0"], busmap_table["0"]):
        bus_name = str(bus_value)
        region_name = str(int(float(region_value)))
        bus_to_region[bus_name] = region_name

    # Load the region shapes and stop early when the file is empty.
    regions_geojson = load_geojson(profile_regions_file)
    if regions_geojson is None:
        raise ValueError("Required GeoJSON is empty: regions_onshore_elec_s.geojson")

    # Index the region features by name so we can look them up quickly later.
    regions_by_name: dict[str, dict] = {}
    for feature in regions_geojson.get("features", []):
        region_name = str(int(float(feature["properties"]["name"])))
        regions_by_name[region_name] = feature

    # Build the output list in the exact order of the incoming bus names.
    ordered_regions: list[dict] = []
    for bus_name in profile_bus_names:
        region_name = bus_to_region[bus_name]
        feature = regions_by_name[region_name]
        geometry = feature["geometry"]
        center_x, center_y = calculate_geometry_center(geometry)

        region_entry = {
            "name": region_name,
            "x": center_x,
            "y": center_y,
            "geometry": geometry,
        }
        ordered_regions.append(region_entry)

    return ordered_regions


def build_regions_geojson(regions: list[dict]) -> dict:
    """Build a minimal GeoJSON object from the region list.

    Args:
        regions: Region dictionaries with names and geometries.

    Returns:
        A GeoJSON feature collection with the same region geometries.
    """
    # Start with an empty feature collection and add one feature per region.
    geojson = {"type": "FeatureCollection", "features": []}

    for region in regions:
        feature = {
            "type": "Feature",
            "properties": {"name": region["name"]},
            "geometry": region["geometry"],
        }
        geojson["features"].append(feature)

    return geojson


def get_geojson_bounds(geojson: dict) -> tuple[float, float, float, float]:
    """Calculate the bounding box of polygon features in a GeoJSON object.

    Args:
        geojson: GeoJSON object containing Polygon or MultiPolygon features.

    Returns:
        A tuple of xmin, ymin, xmax, ymax.
    """
    # Collect all coordinates first and calculate the bounds afterwards.
    x_values: list[float] = []
    y_values: list[float] = []

    for feature in geojson.get("features", []):
        geometry = feature.get("geometry", {})
        geometry_type = geometry.get("type")
        if geometry_type not in {"Polygon", "MultiPolygon"}:
            continue

        for ring_x_values, ring_y_values in iter_polygon_rings(geometry.get("coordinates", [])):
            x_values.extend(ring_x_values)
            y_values.extend(ring_y_values)

    # Fail with a clear message when the geometry does not contain polygons.
    if not x_values or not y_values:
        raise ValueError("GeoJSON does not contain polygon boundaries.")

    # Return the final bounding box as simple min and max values.
    xmin = min(x_values)
    ymin = min(y_values)
    xmax = max(x_values)
    ymax = max(y_values)
    return xmin, ymin, xmax, ymax


def draw_geojson_boundaries(
    ax: plt.Axes,
    geojson: dict,
    color: str,
    linewidth: float,
    zorder: int,
) -> None:
    """Draw polygon boundaries from a GeoJSON object.

    Args:
        ax: Matplotlib axis that receives the boundaries.
        geojson: GeoJSON object containing Polygon or MultiPolygon features.
        color: Line color for the boundaries.
        linewidth: Line width used for plotting.
        zorder: Matplotlib drawing order.

    Returns:
        None.
    """
    # Walk through all polygon features and draw every visible ring.
    for feature in geojson.get("features", []):
        geometry = feature.get("geometry", {})
        geometry_type = geometry.get("type")
        if geometry_type not in {"Polygon", "MultiPolygon"}:
            continue

        for x_values, y_values in iter_polygon_rings(geometry.get("coordinates", [])):
            ax.plot(x_values, y_values, color=color, linewidth=linewidth, zorder=zorder)


def draw_country_shape(ax: plt.Axes, country_shapes_geojson: dict) -> None:
    """Draw the country polygon as a background layer.

    Args:
        ax: Matplotlib axis that receives the country shape.
        country_shapes_geojson: GeoJSON object with the country polygon.

    Returns:
        None.
    """
    # Fill the country polygons first so all other layers stay visible above them.
    for feature in country_shapes_geojson.get("features", []):
        geometry = feature.get("geometry", {})
        geometry_type = geometry.get("type")
        if geometry_type not in {"Polygon", "MultiPolygon"}:
            continue

        for x_values, y_values in iter_polygon_rings(geometry.get("coordinates", [])):
            ax.fill(
                x_values,
                y_values,
                facecolor="#f2efe8",
                edgecolor="#d8d2c4",
                linewidth=0.8,
                zorder=0,
            )

def plot_single_renewable_potential_with_profiles(
    project_paths: ProjectPaths,
    profile_file: Path,
    output_file: Path,
    high_color: str,
) -> Path:
    """Export a renewable potential map with per-region time series panels.

    Args:
        project_paths: Project-specific input and output paths.
        profile_file: Path to the renewable profile file.
        output_file: PNG target path.
        high_color: Main color used for the map and time series.

    Returns:
        The written PNG path.
    """
    # Load all required input data before creating the figure.
    country_shapes_geojson = load_geojson(project_paths.country_shape_file)
    if country_shapes_geojson is None:
        raise ValueError("Required GeoJSON is empty: country_shapes.geojson")

    profile_data = load_renewable_profile_data(profile_file)
    regions = load_profile_regions(
        profile_data["bus"],
        project_paths.profile_busmap_file,
        project_paths.profile_regions_file,
    )

    # Make sure the output folder exists before saving the plot.
    ensure_output_dir(output_file.parent)

    # Sort the regions from north to south and then from west to east.
    sort_order = sorted(
        range(len(regions)),
        key=lambda index: (-regions[index]["y"], regions[index]["x"]),
    )

    sorted_regions: list[dict] = []
    for index in sort_order:
        sorted_regions.append(regions[index])

    sorted_profiles = profile_data["profile"][:, sort_order]
    sorted_capacities_gw = profile_data["p_nom_max"][sort_order] / 1000.0

    # Build the figure layout with one map and one small time series per region.
    fig = plt.figure(figsize=(12, 8))
    grid_spec = fig.add_gridspec(
        nrows=len(sorted_regions),
        ncols=4,
        width_ratios=[1.3, 1.0, 0.10, 0.06],
        wspace=0.08,
        hspace=0.05,
    )
    ax_map = fig.add_subplot(grid_spec[:, 1])
    ax_cbar = fig.add_subplot(grid_spec[:, 3])

    # Draw the country and region boundaries on top of the potential map.
    draw_geojson_boundaries(ax_map, country_shapes_geojson, color="black", linewidth=1.0, zorder=4)
    regions_geojson = build_regions_geojson(sorted_regions)
    draw_geojson_boundaries(ax_map, regions_geojson, color="black", linewidth=0.7, zorder=3)

    # Draw the raster-like potential background with a simple two-color map.
    potential_cmap = LinearSegmentedColormap.from_list(
        f"potential_{profile_file.stem}",
        ["white", high_color],
    )
    potential_grid_gw = profile_data["potential"] / 1000.0
    mappable = ax_map.pcolormesh(
        profile_data["x"],
        profile_data["y"],
        potential_grid_gw,
        shading="auto",
        cmap=potential_cmap,
        alpha=0.6,
        zorder=2,
    )
    colorbar = fig.colorbar(mappable, ax=ax_map, cax=ax_cbar, orientation="vertical")
    colorbar.ax.yaxis.set_ticks_position("right")
    colorbar.ax.yaxis.set_label_position("right")
    colorbar.set_label("Potential [GW]")

    # Fit the map to the country bounds and hide the standard axes.
    xmin, ymin, xmax, ymax = get_geojson_bounds(country_shapes_geojson)
    ax_map.set_xlim(xmin, xmax)
    ax_map.set_ylim(ymin, ymax)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_title("Potential [GW]", pad=4)
    ax_map.set_axis_off()

    # Create one small profile axis for each sorted region.
    profile_axes: list[plt.Axes] = []
    for row_index, capacity_value_gw in enumerate(sorted_capacities_gw):
        if profile_axes:
            shared_axis = profile_axes[0]
        else:
            shared_axis = None

        ax = fig.add_subplot(grid_spec[row_index, 0], sharex=shared_axis)
        ax.plot(profile_data["time"], sorted_profiles[:, row_index], linewidth=1.0, color=high_color)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 1.0])
        ax.set_ylabel(f"{float(capacity_value_gw):.2f} GW", rotation=0, va="center", fontsize=9)
        ax.yaxis.set_label_coords(-0.16, 0.5)
        ax.tick_params(axis="x", labelsize=8)

        # Alternate the 0 and 1 tick labels between the left and right side
        # so stacked axes do not place all numbers on top of each other.
        if row_index % 2 == 0:
            ax.tick_params(axis="y", labelsize=7, pad=1, labelleft=True, labelright=False)
            ax.yaxis.tick_left()
        else:
            ax.tick_params(axis="y", labelsize=7, pad=1, labelleft=False, labelright=True)
            ax.yaxis.tick_right()

        if row_index < len(sorted_regions) - 1:
            ax.label_outer()
        profile_axes.append(ax)

    # Only the last profile panel needs the shared x-axis label.
    profile_axes[-1].set_xlabel("Time")

    # Draw a connection line from each region center to its profile panel.
    for row_index, region in enumerate(sorted_regions):
        connection = ConnectionPatch(
            xyA=(region["x"], region["y"]),
            coordsA=ax_map.transData,
            xyB=(1.0, 0.5),
            coordsB=profile_axes[row_index].transAxes,
            color="gray",
            linewidth=0.8,
            alpha=0.7,
        )
        fig.add_artist(connection)

    # Save the finished figure and free the matplotlib resources.
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_file


def plot_wind_and_pv_potential(project_paths: ProjectPaths) -> tuple[Path, Path]:
    """Export separate potential/profile plots for solar and wind.

    Args:
        project_paths: Project-specific input and output paths.

    Returns:
        The written solar and wind PNG paths.
    """
    # Create the solar plot first.
    solar_output = plot_single_renewable_potential_with_profiles(
        project_paths=project_paths,
        profile_file=project_paths.solar_profile_file,
        output_file=project_paths.solar_potential_output_file,
        high_color="#FDB813",
    )

    # Create the wind plot second.
    wind_output = plot_single_renewable_potential_with_profiles(
        project_paths=project_paths,
        profile_file=project_paths.wind_profile_file,
        output_file=project_paths.wind_potential_output_file,
        high_color="#7FB3D5",
    )
    return solar_output, wind_output

def plot_single_renewable_potential_median_overview(
    project_paths: ProjectPaths,
    profile_file: Path,
    output_file: Path,
    title: str,
    high_color: str,
    show_duration_curves: bool = True,
) -> Path:
    """Export a regional potential map and median time series overview.

    Args:
        project_paths: Project-specific input and output paths.
        profile_file: Path to the renewable profile file.
        output_file: PNG target path.
        title: Technology name shown in plot titles.
        high_color: Main color used for the map and time series.

    Returns:
        The written PNG path.
    """
    # Prepare the output folder and read the profile inputs.
    ensure_output_dir(output_file.parent)
    profile_data = load_renewable_profile_data(profile_file)
    regions = load_profile_regions(
        profile_data["bus"],
        project_paths.profile_busmap_file,
        project_paths.profile_regions_file,
    )

    # Create one map axis on the left and two stacked profile axes on the right.
    if show_duration_curves:
        fig = plt.figure(figsize=(16, 9))
        grid_spec = fig.add_gridspec(
            nrows=2,
            ncols=2,
            width_ratios=[1.0, 1.35],
            height_ratios=[1.0, 1.0],
            wspace=0.18,
            hspace=0.28,
        )

        ax_map = fig.add_subplot(grid_spec[:, 0])
        ax_time = fig.add_subplot(grid_spec[0, 1])
        ax_duration = fig.add_subplot(grid_spec[1, 1])

    else:
        fig = plt.figure(figsize=(16, 7))
        grid_spec = fig.add_gridspec(
            nrows=1,
            ncols=2,
            width_ratios=[1.0, 1.35],
            wspace=0.18,
        )

        ax_map = fig.add_subplot(grid_spec[0, 0])
        ax_time = fig.add_subplot(grid_spec[0, 1])
        ax_duration = None

    # Convert the regional capacities from MW to GW for the map labels and colors.
    region_values_gw = profile_data["p_nom_max"] / 1000.0
    cmap = LinearSegmentedColormap.from_list(f"gw_{title}", ["white", high_color])
    norm = Normalize(vmin=float(region_values_gw.min()), vmax=float(region_values_gw.max()))

    # Draw each region with its own fill color and place the value in the region center.
    for region, value_gw in zip(regions, region_values_gw):
        geometry = region["geometry"]
        face_color = cmap(norm(float(value_gw)))

        for x_values, y_values in iter_polygon_rings(geometry["coordinates"]):
            ax_map.fill(
                x_values,
                y_values,
                facecolor=face_color,
                edgecolor="white",
                linewidth=0.7,
                zorder=2,
            )

        polygon = shape(region["geometry"])
        label_point = polygon.representative_point()

        if project_paths.country_name == "Bulgaria":
            label_offsets = {
                "258": (0.0, -0.1),
                "177": (0.0, -0.1),
                "51": (0.0, -0.1),
                "46": (0.0, +0.1),
            }
        elif project_paths.country_name == "Romania":
            label_offsets = {
                "50": (0.0, -0.050),
                "2": (-0.01, 0.13),
                "4": (-0.02, 0.19),
                "116": (0.0, -0.1),
            }
        else:
            label_offsets = {}

        x_offset, y_offset = label_offsets.get(region["name"], (0.0, 0.0))

        ax_map.text(
            label_point.x + x_offset,
            label_point.y + y_offset,
            f"{value_gw:.1f}",
            fontsize=7,
            color="black",
            ha="center",
            va="center",
            zorder=4,
        )

    # Draw the region borders after the filled polygons so the map stays sharp.
    regions_geojson = build_regions_geojson(regions)
    draw_geojson_boundaries(ax_map, regions_geojson, color="black", linewidth=0.7, zorder=3)

    # Fit the map to the visible region bounds and add the colorbar.
    xmin, ymin, xmax, ymax = get_geojson_bounds(regions_geojson)
    ax_map.set_xlim(xmin, xmax)
    ax_map.set_ylim(ymin, ymax)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_axis_off()
    ax_map.set_title(f"{title}: Regional Potential [GW]")

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=ax_map, fraction=0.046, pad=0.02)
    colorbar.set_label("Potential [GW]")

    # Build the median profile from all regional time series.
    profile_matrix = np.array(profile_data["profile"], dtype=float)
    median_profile = np.median(profile_matrix, axis=1)

    if show_duration_curves:
        # Determine the time step in hours so FLH values stay correct even if the data changes.
        time_step_hours = 1.0
        if len(profile_data["time"]) > 1:
            first_step = profile_data["time"][1] - profile_data["time"][0]
            time_step_hours = float(first_step / pd.Timedelta(hours=1))
            if time_step_hours <= 0.0:
                time_step_hours = 1.0

        # Calculate the FLH for every region and identify the lowest and highest cases.
        region_flh_hours = np.sum(profile_matrix, axis=0) * time_step_hours
        min_flh_index = int(np.argmin(region_flh_hours))
        max_flh_index = int(np.argmax(region_flh_hours))

        # Convert every regional profile into one annual duration curve.
        duration_curves: list[np.ndarray] = []
        for column_index in range(profile_matrix.shape[1]):
            region_curve = np.sort(profile_matrix[:, column_index])[::-1]
            duration_curves.append(region_curve)

        min_duration_curve = duration_curves[min_flh_index]
        max_duration_curve = duration_curves[max_flh_index]
        duration_hours = np.arange(1, len(duration_curves[0]) + 1, dtype=float) * time_step_hours

    # Draw the median time series in the top-right panel.
    ax_time.plot(profile_data["time"], median_profile, color=high_color, linewidth=1.8)
    ax_time.set_title(f"{title}: Median Relative Profile")
    ax_time.set_xlabel("Time")
    ax_time.set_ylabel("Profile [-]")
    ax_time.set_ylim(0.0, 1.0)
    ax_time.set_xlim(profile_data["time"][0], profile_data["time"][-1])
    ax_time.margins(x=0.0)
    ax_time.grid(alpha=0.2)
    ax_time.xaxis.set_major_locator(mdates.MonthLocator())
    ax_time.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    if show_duration_curves:
        # Fill the area between the lowest and highest FLH curves.
        ax_duration.fill_between(
            duration_hours,
            min_duration_curve,
            max_duration_curve,
            color=high_color,
            alpha=0.12,
            zorder=1,
        )

        # Draw all regional duration curves as thin lines so the spread becomes visible.
        flh_order = np.argsort(region_flh_hours)
        min_line_color = np.array([0.60, 0.60, 0.60])
        max_line_color = np.array(matplotlib.colors.to_rgb(high_color))
        for rank, region_index in enumerate(flh_order):
            if len(flh_order) == 1:
                blend_factor = 1.0
            else:
                blend_factor = rank / (len(flh_order) - 1)

            region_color = min_line_color + blend_factor * (max_line_color - min_line_color)
            ax_duration.plot(
                duration_hours,
                duration_curves[int(region_index)],
                color=region_color,
                linewidth=0.6,
                alpha=0.32,
                zorder=2,
            )

        # Highlight the lowest and highest FLH curves explicitly.
        ax_duration.plot(
            duration_hours,
            min_duration_curve,
            color="#8f8f8f",
            linewidth=1.5,
            linestyle="--",
            label=f"Lowest FLH ({region_flh_hours[min_flh_index]:.0f} h)",
            zorder=3,
        )
        ax_duration.plot(
            duration_hours,
            max_duration_curve,
            color=high_color,
            linewidth=1.7,
            linestyle="-",
            label=f"Highest FLH ({region_flh_hours[max_flh_index]:.0f} h)",
            zorder=4,
        )

        # Mark the lowest and highest FLH values as simple vertical guide lines.
        min_flh_hours = float(region_flh_hours[min_flh_index])
        max_flh_hours = float(region_flh_hours[max_flh_index])
        ax_duration.axvline(min_flh_hours, color="#8f8f8f", linewidth=1.0, linestyle="--", zorder=2.5)
        ax_duration.axvline(max_flh_hours, color=high_color, linewidth=1.0, linestyle="-", zorder=2.6)

        ax_duration.set_title(f"{title}: Annual Duration Curves")
        ax_duration.set_xlabel("Sorted hours of the year")
        ax_duration.set_ylabel("Profile [-]")
        ax_duration.set_xlim(0.0, float(duration_hours[-1]))
        ax_duration.set_ylim(0.0, 1.0)
        ax_duration.grid(alpha=0.2)
        ax_duration.legend(loc="upper right", fontsize=9)

    # Finish the figure styling and save the result.
    fig.autofmt_xdate(rotation=45)
    fig.suptitle(f"{project_paths.country_name} - {title} Potential Overview", fontsize=20)
    if show_duration_curves:
        fig.subplots_adjust(
            left=0.05,
            right=0.98,
            bottom=0.08,
            top=0.86,
            wspace=0.22,
            hspace=0.30,
        )
    else:
        fig.subplots_adjust(
            left=0.05,
            right=0.98,
            bottom=0.10,
            top=0.86,
            wspace=0.22,
        )
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_file


def plot_wind_and_pv_potential_median_overview(
    project_paths: ProjectPaths,
) -> tuple[Path, Path]:
    """Export median overview plots for solar and wind.

    Args:
        project_paths: Project-specific input and output paths.

    Returns:
        The written solar and wind overview PNG paths.
    """
    # Create the PV overview first.
    solar_output = plot_single_renewable_potential_median_overview(
        project_paths=project_paths,
        profile_file=project_paths.solar_profile_file,
        output_file=project_paths.solar_potential_median_output_file,
        title="PV",
        high_color="#FDB813",
        show_duration_curves=show_duration_curves
    )

    # Create the wind overview second.
    wind_output = plot_single_renewable_potential_median_overview(
        project_paths=project_paths,
        profile_file=project_paths.wind_profile_file,
        output_file=project_paths.wind_potential_median_output_file,
        title="Wind",
        high_color="#7FB3D5",
        show_duration_curves=show_duration_curves
    )
    return solar_output, wind_output


def main() -> None:
    """Generate all plot exports for every valid country project under data.

    Args:
        None.

    Returns:
        None.
    """
    if not project_dir.exists():
        raise FileNotFoundError(project_dir)

    resources_dir = project_dir / "resources" / scenario
    network_dir = project_dir / "results" / scenario / "postnetworks"

    matching_network_files = sorted(network_dir.glob(RESULT_NETWORK_FILE_PATTERN))

    if not resources_dir.exists():
        raise FileNotFoundError(resources_dir)

    if len(matching_network_files) != 1:
        raise ValueError(
            f"Expected exactly one result network in {network_dir}, "
            f"found {len(matching_network_files)}."
        )

    result_network_file = matching_network_files[0]

    # Build one explicit path object for the current country.
    project_paths = ProjectPaths(
        country_name=project_dir.parent.name,
        base_network_dir=resources_dir / "base_network",
        country_shape_file=resources_dir / "shapes" / "country_shapes.geojson",
        result_network_file=result_network_file,
        solar_profile_file=resources_dir / "renewable_profiles" / "profile_solar.nc",
        wind_profile_file=resources_dir / "renewable_profiles" / "profile_onwind.nc",
        profile_busmap_file=resources_dir / "bus_regions" / "busmap_elec_s.csv",
        profile_regions_file=resources_dir / "bus_regions" / "regions_onshore_elec_s.geojson",
        figure_dir=figure_dir,
        base_network_output_file=figure_dir / "base_network_overview.png",
        result_network_output_file=figure_dir / "result_network_overview.png",
        result_network_labeled_output_file=figure_dir / "result_network_overview_labeled.png",
        solar_potential_output_file=figure_dir / "solar_potential_with_profiles.png",
        wind_potential_output_file=figure_dir / "wind_potential_with_profiles.png",
        solar_potential_median_output_file=figure_dir / "solar_potential_median_overview.png",
        wind_potential_median_output_file=figure_dir / "wind_potential_median_overview.png",
    )

    # Run all exports for the current country.
    plot_wind_and_pv_potential(project_paths)
    plot_wind_and_pv_potential_median_overview(project_paths)

    # Print the output folder so the finished run is easy to verify.
    print(f"Finished: {figure_dir}")


if __name__ == "__main__":
    main()
