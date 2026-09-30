"""Prepare small spatial dashboard bundles from solved PyPSA-Earth networks."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import xarray as xr


METRICS = {
    'hydrogen_production': ('Hydrogen production potential', 'TWh H₂', 'Total hydrogen production based on available electrolyser capacity, infrastructure and renewable energy supply.'),
    'hydrogen_export': ('Hydrogen export potential', 'TWh H₂', 'Total hydrogen export, attributed to the respective region.'),
    'electrolyser_capacity': ('Electrolyser capacity', 'GWₑ', 'Optimised electrical input capacity for hydrogen production.'),
    'hydrogen_storage': ('Hydrogen storage capacity', 'GWh H₂', 'Optimised energy capacity of stores connected to regional H₂ buses.'),
    'solar_capacity': ('Solar capacity', 'GWₑ', 'Optimised solar generator capacity.'),
    'wind_capacity': ('Onshore wind capacity', 'GWₑ', 'Optimised onshore wind generator capacity.'),
    'solar_potential': ('Solar capacity potential', 'GWₑ', 'Finite p_nom_max limits of solar generators; a capacity limit, not annual H₂ potential.'),
    'wind_potential': ('Onshore wind capacity potential', 'GWₑ', 'Finite p_nom_max limits of onshore wind generators; a capacity limit, not annual H₂ potential.'),
}


def aggregate(values, locations, regions):
    if values.empty:
        return pd.Series(0.0, index=regions)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite values in a regional metric.')
    if locations.isna().any() or not locations.isin(regions).all():
        raise ValueError('A model component cannot be assigned to a regional bus.')
    return values.clip(lower=0).groupby(locations).sum().reindex(regions, fill_value=0)


def weighted_output(flow, weights):
    """PyPSA output ports are negative for delivered power; preserve snapshot hours."""
    if not flow.index.equals(weights.index) or not np.isfinite(weights).all() or weights.lt(0).any():
        raise ValueError('Invalid or misaligned snapshot weights.')
    if flow.isna().any().any():
        raise ValueError('Missing solved output time series.')
    return (-flow).clip(lower=0).mul(weights, axis=0).sum() / 1e6


def prepare_network(network_path, regions_path, output_dir, country, run):
    import pypsa

    network_path, regions_path, output_dir = map(Path, (network_path, regions_path, output_dir))
    n = pypsa.Network(network_path)
    if n.links_t.p1.empty:
        raise ValueError(f'{network_path} has no solved link flows; use results/.../postnetworks.')
    geo = json.loads(regions_path.read_text())
    regions = pd.Index([f['properties']['name'] for f in geo['features']], name='region')
    if regions.has_duplicates or not regions.isin(n.buses.index).all():
        raise ValueError('Region boundaries must uniquely match the solved electricity buses.')
    buses = n.buses.loc[regions]
    if not buses.carrier.isin(['AC', 'DC']).all():
        raise ValueError('Expected electricity bus regions.')
    locations = n.buses.location.where(n.buses.location.ne(''))
    locations.loc[regions] = regions
    input_carrier = n.links.bus0.map(n.buses.carrier)
    output_carrier = n.links.bus1.map(n.buses.carrier)
    # Select by ports, including PEM, alkaline variants and SOEC, rather than one carrier label.
    electrolysers = n.links[input_carrier.isin(['AC', 'DC']) & output_carrier.eq('H2')]
    export_buses = n.buses.index[n.buses.index.str.contains(r'H2 export', case=False)]
    exports = n.links[input_carrier.eq('H2') & n.links.bus1.isin(export_buses)]
    stores = n.stores[n.stores.bus.map(n.buses.carrier).eq('H2') & n.stores.bus.map(locations).isin(regions)]
    weights = n.snapshot_weightings.generators
    report = pd.DataFrame(index=regions)
    report['hydrogen_production'] = aggregate(weighted_output(n.links_t.p1[electrolysers.index], weights), electrolysers.bus1.map(locations), regions)
    report['hydrogen_export'] = aggregate(weighted_output(n.links_t.p1[exports.index], weights), exports.bus0.map(locations), regions)
    report['electrolyser_capacity'] = aggregate(electrolysers.p_nom_opt / 1e3, electrolysers.bus0.map(locations), regions)
    report['hydrogen_storage'] = aggregate(stores.e_nom_opt / 1e3, stores.bus.map(locations), regions)
    for technology, carrier in [('solar', 'solar'), ('wind', 'onwind')]:
        generators = n.generators[n.generators.carrier.eq(carrier)]
        report[f'{technology}_capacity'] = aggregate(generators.p_nom_opt / 1e3, generators.bus.map(locations), regions)
        report[f'{technology}_potential'] = aggregate(generators.p_nom_max / 1e3, generators.bus.map(locations), regions)

    records = []
    for frame, carrier, capacity, original in [
        (n.lines, 'AC', 's_nom_opt', 's_nom'),
        (n.links[n.links.carrier.eq('DC')], 'DC', 'p_nom_opt', 'p_nom'),
        (n.links[n.links.carrier.str.startswith('H2 pipeline')], 'H2', 'p_nom_opt', 'p_nom'),
    ]:
        for name, row in frame.iterrows():
            bus0, bus1 = locations.get(row.bus0), locations.get(row.bus1)
            if bus0 not in regions or bus1 not in regions:
                print(f'Warning: Skipping connection {name}: unmatched region for {row.bus0} -> {row.bus1}.')
                continue
            if bus0 == bus1:
                continue
            records.append(dict(name=name, bus0=bus0, bus1=bus1, carrier=carrier,
                                technology=row.carrier, capacity_gw=max(0, row[capacity]) / 1e3,
                                original_capacity_gw=max(0, row[original]) / 1e3))
    edges = pd.DataFrame(records, columns=['name', 'bus0', 'bus1', 'carrier', 'technology', 'capacity_gw', 'original_capacity_gw'])
    ds = report.to_xarray().assign_coords(
        latitude=('region', buses.y.to_numpy()), longitude=('region', buses.x.to_numpy()),
        country=('region', buses.country.fillna(country).to_numpy().astype(str)),
    )
    for key, (label, unit, definition) in METRICS.items():
        ds[key].attrs.update(label=label, units=unit, definition=definition)
    ds.attrs.update(country=country, run=run, source_network=network_path.name,
                    represented_hours=float(weights.sum()),
                    energy_note='Energy totals cover the weighted model period; they are annual only when weights represent a full year.')
    # Validate everything before writing the bundle; the manifest is written last.
    if not np.isfinite(report.to_numpy()).all() or not np.isfinite(buses[['x', 'y']].to_numpy()).all():
        raise ValueError('Non-finite spatial data.')
    output_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_dir / 'report.nc')
    (output_dir / 'regions.geojson').write_text(json.dumps(geo))
    edges.to_csv(output_dir / 'network.csv', index=False)
    metadata = dict(country=country, run=run, network=network_path.name,
                    source_network=str(network_path.resolve()), source_regions=str(regions_path.resolve()),
                    region_count=len(regions), edge_count=len(edges), represented_hours=float(weights.sum()))
    (output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def main(argv=None):
    workspace = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir', type=Path, default=workspace / 'Romania' / 'pypsa-earth')
    parser.add_argument('--run', default='1h-sec')
    parser.add_argument('--network', type=Path, help='One solved network; otherwise process all postnetworks in the run.')
    parser.add_argument('--regions', type=Path, help='Override the matching bus-region GeoJSON.')
    parser.add_argument('--output-dir', type=Path, default=workspace / 'HySEE-Streamlit' / 'data' / 'spatial')
    args = parser.parse_args(argv)
    country = args.project_dir.resolve().parent.name
    networks = [args.network] if args.network else sorted((args.project_dir / 'results' / args.run / 'postnetworks').glob('*.nc'))
    if not networks:
        parser.error('No solved networks found in results/<run>/postnetworks.')
    for network in networks:
        match = re.search(r'elec_s_(\d+)_', network.name)
        if args.regions is None and match is None:
            parser.error('Cannot infer region resolution; supply --regions.')
        regions = args.regions or args.project_dir / 'resources' / args.run / 'bus_regions' / f'regions_onshore_elec_s_{match[1]}.geojson'
        destination = args.output_dir / country / args.run / network.stem
        metadata = prepare_network(network, regions, destination, country, args.run)
        print(f'{destination}: {metadata["region_count"]} regions, {metadata["edge_count"]} connections')


if __name__ == '__main__':
    main()
