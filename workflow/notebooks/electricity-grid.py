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


def plot_base_network_overview(project_paths: ProjectPaths) -> Path:
    """Export the base network topology plot.

    Args:
        project_paths: Project-specific input and output paths.

    Returns:
        The written PNG path.
    """
    # Load the background country shape and the line layers.
    country_shapes_geojson = load_geojson(project_paths.country_shape_file)
    lines_geojson = load_geojson(project_paths.base_network_dir / "all_lines_build_network.geojson")
    converters_geojson = load_geojson(
        project_paths.base_network_dir / "all_converters_build_network.geojson"
    )

    if country_shapes_geojson is None:
        raise ValueError("Required GeoJSON is empty: country_shapes.geojson")
    if lines_geojson is None:
        raise ValueError("Required GeoJSON is empty: all_lines_build_network.geojson")

    # Prepare the output folder and the base figure.
    ensure_output_dir(project_paths.base_network_output_file.parent)
    fig, ax = plt.subplots(figsize=(10, 10))
    draw_country_shape(ax, country_shapes_geojson)

    # Draw all transmission lines with voltage-based colors and AC/DC styles.
    for feature in lines_geojson.get("features", []):
        geometry = feature.get("geometry", {})
        geometry_type = geometry.get("type")
        if geometry_type not in {"LineString", "MultiLineString"}:
            continue

        properties = feature.get("properties", {})
        voltage_value = properties.get("voltage")
        try:
            voltage_key = int(float(voltage_value))
            line_color = VOLTAGE_COLORS.get(voltage_key, "#8f8f8f")
        except (TypeError, ValueError):
            line_color = "#8f8f8f"

        dc_value = properties.get("dc", False)
        if isinstance(dc_value, str):
            is_dc = dc_value.strip().lower() in {"true", "1", "yes"}
        else:
            is_dc = bool(dc_value)

        for x_values, y_values in iter_line_segments(geometry.get("coordinates", [])):
            ax.plot(
                x_values,
                y_values,
                color=line_color,
                linewidth=0.8,
                alpha=0.85,
                linestyle="--" if is_dc else "-",
                zorder=1,
            )

    # Draw optional converters as a separate dashed purple layer.
    if converters_geojson is not None:
        for feature in converters_geojson.get("features", []):
            geometry = feature.get("geometry", {})
            geometry_type = geometry.get("type")
            if geometry_type not in {"LineString", "MultiLineString"}:
                continue

            for x_values, y_values in iter_line_segments(geometry.get("coordinates", [])):
                ax.plot(
                    x_values,
                    y_values,
                    color="#7b4cc2",
                    linewidth=1.4,
                    alpha=0.9,
                    linestyle="--",
                    zorder=6,
                )

    # Build the legend in a very explicit beginner-friendly way.
    legend_handles = [
        Line2D([], [], color="#d8d2c4", linewidth=2.0, label="Country shape"),
        Line2D([], [], color="#666666", linewidth=1.2, linestyle="-", label="AC"),
        Line2D([], [], color="#666666", linewidth=1.2, linestyle="--", label="DC"),
        Line2D([], [], color=VOLTAGE_COLORS[110000], linewidth=1.6, label="110 kV"),
        Line2D([], [], color=VOLTAGE_COLORS[220000], linewidth=1.6, label="220 kV"),
        Line2D([], [], color=VOLTAGE_COLORS[400000], linewidth=1.6, label="400 kV"),
    ]

    if converters_geojson is not None and converters_geojson.get("features"):
        converter_handle = Line2D(
            [],
            [],
            color="#7b4cc2",
            linewidth=1.4,
            linestyle="--",
            label="Converters",
        )
        legend_handles.append(converter_handle)

    # Apply the final plot styling and save the image.
    ax.set_title(f"{project_paths.country_name} - Electricity Grid Topology")
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.margins(0.02)
    ax.legend(handles=legend_handles, loc="lower right", frameon=True)

    fig.savefig(project_paths.base_network_output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return project_paths.base_network_output_file


def export_result_network_tables(project_paths: ProjectPaths) -> tuple[Path, Path]:
    """Export result network lines and links as CSV tables.

    Args:
        project_paths: Project-specific input and output paths.

    Returns:
        The written lines CSV path and links CSV path.
    """
    # Read the shared network datasets once so both tables use the same source data.
    network_data = load_result_network_data(project_paths.result_network_file)

    # Build a lookup from bus name to x/y coordinates for the exported endpoint columns.
    bus_positions: dict[str, tuple[float, float]] = {}
    for name, x_value, y_value in zip(
        network_data["bus_names"],
        network_data["bus_x_values"],
        network_data["bus_y_values"],
    ):
        bus_positions[name] = (x_value, y_value)

    # Build the lines table row by row so every exported column stays obvious.
    line_rows: list[dict] = []
    for name, bus0_name, bus1_name, carrier_name, length_km, voltage_v, s_nom_mva in zip(
        network_data["line_names"],
        network_data["line_bus0"],
        network_data["line_bus1"],
        network_data["line_carriers"],
        network_data["line_lengths"],
        network_data["line_v_nom"],
        network_data["line_s_nom"],
    ):
        bus0_position = bus_positions[bus0_name]
        bus1_position = bus_positions[bus1_name]
        line_rows.append(
            {
                "name": name,
                "bus0": bus0_name,
                "bus1": bus1_name,
                "carrier": carrier_name,
                "length_km": length_km,
                "v_nom_v": voltage_v,
                "s_nom_mva": s_nom_mva,
                "x0": bus0_position[0],
                "y0": bus0_position[1],
                "x1": bus1_position[0],
                "y1": bus1_position[1],
            }
        )

    # Build the links table with the same explicit endpoint coordinates.
    link_rows: list[dict] = []
    for name, bus0_name, bus1_name, carrier_name, p_nom_opt_mw in zip(
        network_data["link_names"],
        network_data["link_bus0"],
        network_data["link_bus1"],
        network_data["link_carriers"],
        network_data["link_p_nom_opt"],
    ):
        bus0_position = bus_positions[bus0_name]
        bus1_position = bus_positions[bus1_name]
        link_rows.append(
            {
                "name": name,
                "bus0": bus0_name,
                "bus1": bus1_name,
                "carrier": carrier_name,
                "p_nom_opt_mw": p_nom_opt_mw,
                "x0": bus0_position[0],
                "y0": bus0_position[1],
                "x1": bus1_position[0],
                "y1": bus1_position[1],
            }
        )

    # Write both CSV files in a plain pandas table format.
    lines_output_file = table_dir / "result_network_lines.csv"
    links_output_file = table_dir / "result_network_links.csv"
    pd.DataFrame(line_rows).to_csv(lines_output_file, index=False)
    pd.DataFrame(link_rows).to_csv(links_output_file, index=False)

    return lines_output_file, links_output_file

def plot_result_network_overview(project_paths: ProjectPaths) -> Path:
    """Export result network plots with and without labels.

    Args:
        project_paths: Project-specific input and output paths.

    Returns:
        The written unlabeled PNG path.
    """
    # Load the background country geometry first.
    country_shapes_geojson = load_geojson(project_paths.country_shape_file)
    if country_shapes_geojson is None:
        raise ValueError("Required GeoJSON is empty: country_shapes.geojson")

    # Make sure the output folder exists before saving images.
    ensure_output_dir(project_paths.result_network_output_file.parent)

    # Read all required node and line datasets from the network file.
    network_data = load_result_network_data(project_paths.result_network_file)
    bus_names = network_data["bus_names"]
    bus_x_values = network_data["bus_x_values"]
    bus_y_values = network_data["bus_y_values"]
    line_names = network_data["line_names"]
    line_bus0 = network_data["line_bus0"]
    line_bus1 = network_data["line_bus1"]
    line_carriers = network_data["line_carriers"]
    link_names = network_data["link_names"]
    link_bus0 = network_data["link_bus0"]
    link_bus1 = network_data["link_bus1"]
    link_carriers = network_data["link_carriers"]

    # Build a simple lookup table for node coordinates.
    bus_positions: dict[str, tuple[float, float]] = {}
    for name, x_value, y_value in zip(bus_names, bus_x_values, bus_y_values):
        bus_positions[name] = (x_value, y_value)

    # Collect all AC lines and remember which nodes belong to them.
    ac_lines: list[dict] = []
    ac_node_names: set[str] = set()
    for line_name, bus0_name, bus1_name, carrier_name in zip(
        line_names, line_bus0, line_bus1, line_carriers
    ):
        if carrier_name.upper() != "AC":
            continue

        ac_node_names.add(bus0_name)
        ac_node_names.add(bus1_name)
        bus0_position = bus_positions[bus0_name]
        bus1_position = bus_positions[bus1_name]

        line_entry = {
            "name": line_name,
            "x0": bus0_position[0],
            "y0": bus0_position[1],
            "x1": bus1_position[0],
            "y1": bus1_position[1],
        }
        ac_lines.append(line_entry)

    # Collect all DC links and remember which nodes belong to them.
    dc_lines: list[dict] = []
    dc_node_names: set[str] = set()
    for link_name, bus0_name, bus1_name, carrier_name in zip(
        link_names, link_bus0, link_bus1, link_carriers
    ):
        if "dc" not in carrier_name.lower():
            continue

        dc_node_names.add(bus0_name)
        dc_node_names.add(bus1_name)
        bus0_position = bus_positions[bus0_name]
        bus1_position = bus_positions[bus1_name]

        link_entry = {
            "name": link_name,
            "x0": bus0_position[0],
            "y0": bus0_position[1],
            "x1": bus1_position[0],
            "y1": bus1_position[1],
        }
        dc_lines.append(link_entry)

    # Build explicit node lists for the scatter plots.
    ac_nodes: list[dict] = []
    for node_name in sorted(ac_node_names):
        position = bus_positions[node_name]
        ac_nodes.append({"name": node_name, "x": position[0], "y": position[1]})

    dc_nodes: list[dict] = []
    for node_name in sorted(dc_node_names):
        position = bus_positions[node_name]
        dc_nodes.append({"name": node_name, "x": position[0], "y": position[1]})

    # Export one plot without labels and one plot with labels.
    target_variants = [
        (project_paths.result_network_output_file, False),
        (project_paths.result_network_labeled_output_file, True),
    ]

    for target_file, show_labels in target_variants:
        fig, ax = plt.subplots(figsize=(10, 10))
        draw_country_shape(ax, country_shapes_geojson)

        # Draw all AC lines and optionally label them in the middle.
        for line in ac_lines:
            ax.plot(
                [line["x0"], line["x1"]],
                [line["y0"], line["y1"]],
                color="#8f8f8f",
                linewidth=0.7,
                alpha=0.85,
                zorder=1,
            )

            if show_labels:
                line_x_mid = (line["x0"] + line["x1"]) / 2
                line_y_mid = (line["y0"] + line["y1"]) / 2
                ax.text(
                    line_x_mid,
                    line_y_mid,
                    line["name"],
                    fontsize=5,
                    color="#5a5a5a",
                    ha="center",
                    va="center",
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.6,
                        "edgecolor": "none",
                        "pad": 0.5,
                    },
                    zorder=6,
                )

        # Draw all DC links and optionally label them in the middle.
        for line in dc_lines:
            ax.plot(
                [line["x0"], line["x1"]],
                [line["y0"], line["y1"]],
                color="#7b4cc2",
                linewidth=1.1,
                alpha=0.9,
                linestyle="--",
                zorder=2,
            )

            if show_labels:
                line_x_mid = (line["x0"] + line["x1"]) / 2
                line_y_mid = (line["y0"] + line["y1"]) / 2
                ax.text(
                    line_x_mid,
                    line_y_mid,
                    line["name"],
                    fontsize=5,
                    color="#7b4cc2",
                    ha="center",
                    va="center",
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.6,
                        "edgecolor": "none",
                        "pad": 0.5,
                    },
                    zorder=6,
                )

        # Draw the AC and DC nodes as separate scatter layers.
        ac_node_x: list[float] = []
        ac_node_y: list[float] = []
        for node in ac_nodes:
            ac_node_x.append(node["x"])
            ac_node_y.append(node["y"])

        ax.scatter(
            ac_node_x,
            ac_node_y,
            s=18,
            color="#1f77b4",
            alpha=0.95,
            linewidths=0,
            zorder=4,
        )

        if dc_nodes:
            dc_node_x: list[float] = []
            dc_node_y: list[float] = []
            for node in dc_nodes:
                dc_node_x.append(node["x"])
                dc_node_y.append(node["y"])

            ax.scatter(
                dc_node_x,
                dc_node_y,
                s=20,
                color="#7b4cc2",
                alpha=0.95,
                linewidths=0,
                zorder=5,
            )

        # Add node names only to the labeled export.
        if show_labels:
            for node in ac_nodes:
                ax.text(
                    node["x"],
                    node["y"],
                    node["name"],
                    fontsize=6,
                    color="#1f77b4",
                    ha="left",
                    va="bottom",
                    zorder=7,
                )

            for node in dc_nodes:
                ax.text(
                    node["x"],
                    node["y"],
                    node["name"],
                    fontsize=6,
                    color="#7b4cc2",
                    ha="left",
                    va="bottom",
                    zorder=7,
                )

        # Build the legend items one by one for clarity.
        legend_handles = [
            Line2D([], [], color="#d8d2c4", linewidth=2.0, label="Country shape"),
            Line2D([], [], color="#8f8f8f", linewidth=1.4, label="AC lines"),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                markersize=6,
                markerfacecolor="#1f77b4",
                markeredgecolor="#1f77b4",
                label="AC nodes",
            ),
        ]

        if dc_lines:
            legend_handles.append(
                Line2D(
                    [],
                    [],
                    color="#7b4cc2",
                    linewidth=1.4,
                    linestyle="--",
                    label="DC links",
                )
            )
            legend_handles.append(
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="None",
                    markersize=6,
                    markerfacecolor="#7b4cc2",
                    markeredgecolor="#7b4cc2",
                    label="DC nodes",
                )
            )

        # Apply the final styling and save the current variant.
        ax.set_title(f"{project_paths.country_name} Result Network Overview")
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.margins(0.02)
        ax.legend(handles=legend_handles, loc="lower right", frameon=True)

        fig.savefig(target_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return project_paths.result_network_output_file


def main() -> None:
    """Generate all plots for a single PyPSA-Earth project."""

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

    plot_base_network_overview(project_paths)
    plot_result_network_overview(project_paths)
    export_result_network_tables(project_paths)

    print(f"Finished: {figure_dir}")


if __name__ == "__main__":
    main()
