# This script prepares results from PyPSA-Eur for Streamlit visualization
# Author: Alexander Meisinger
# Project: H2Global meets Africa (FENES, OTH Regensburg)
# Base: https://doi.org/10.1016/j.joule.2023.06.016 and https://github.com/PyPSA/pypsa-eur

# Package imports
import os
import re
import sys
import yaml
import pandas as pd
import numpy as np
import xarray as xr
import pypsa

# Configure global settings
xr.set_options(display_style="html")


def capacity(scenarios, country):
    """
    Process installed and optimal energy system capacities from PyPSA-Eur NetCDF scenario files.

    This function reads a set of scenario-specific PyPSA network files, extracts optimal capacities,
    and returns electricity generation, conversion, and storage capacities.

    Parameters:
        scenarios (str): Path to the scenario directory containing `configs/` and `networks/`.
        country (str): Either a specific country code (e.g., "DE") or "EU" for full-European analysis.

    Returns:
        gen (pd.DataFrame): Installed generation and storage capacity in GW per carrier.
        con (pd.DataFrame): Installed conversion capacity in GW per carrier (e.g. links).
        twh (pd.DataFrame): Annual energy output in TWh per carrier (excluding generators and links).

    Notes:
        - Converts MWh to TWh and MW to GW for interpretability.
        - Drops unused scenario variations based on fixed config values.
        - Filters country-specific data if `country != "EU"`.
    """
    # Read config settings for scenario
    files = sorted([f for f in os.listdir(f"{scenarios}/configs") if os.path.isfile(os.path.join(f"{scenarios}/configs", f))])
    if files:
        with open(os.path.join(f"{scenarios}/configs", files[0]), 'r') as file:
            config = yaml.safe_load(file)

    # Get results from scenario (nc file)
    networks_dict = {
        (cluster, ll, opt + sector_opt, planning_horizon): f"{scenarios}"
        + f"/networks/base_s_{cluster}_{opt}_{sector_opt}_{planning_horizon}.nc"
        for cluster in config["scenario"]["clusters"]
        for opt in config["scenario"]["opts"]
        for sector_opt in config["scenario"]["sector_opts"]
        for ll in [config["electricity"]["transmission_limit"]]
        for planning_horizon in config["scenario"]["planning_horizons"]
    }

    # Set sceanrio settings in dataset
    columns = pd.MultiIndex.from_tuples(
        networks_dict.keys(),
        names=["clusters", "ll", "opt", "planning_horizon"],
    )
    cap = pd.DataFrame(columns=columns, dtype=float)
    print(f"Capacity analysis for {country} and scenario {scenarios}")
    # Set results in dataset
    for label, filename in networks_dict.items():
        # Read nc file
        n = pypsa.Network(filename)
        # Extract and sort results from nc file
        if country == "EU":
            df_cap_annual = pd.concat([n.statistics.optimal_capacity()])
        else:
            df_cap_annual = pd.concat([n.statistics.optimal_capacity(groupby=["carrier", "country"])])
            idx = pd.IndexSlice
            df_cap_annual = df_cap_annual.loc[idx[:,:,country]]
            
            df_cap_annual = df_cap_annual.reorder_levels(['component', 'carrier'])
            df_cap_annual = df_cap_annual.sort_index()

        cap = cap.reindex(df_cap_annual.index.union(cap.index))
        cap.loc[df_cap_annual.index, label] = df_cap_annual
    # Fill nan values
    df = cap.fillna(0)
    # Drop non relevant scenario informations
    df = df.xs((config["scenario"]["clusters"][0], config["electricity"]["transmission_limit"], config["scenario"]["sector_opts"][0]), level=["clusters", "ll", "opt"], axis=1)
    # Sort results
    twh_columns_to_drop = ["Generator", "Link", "Line", "StorageUnit"]
    twh_existing_columns = [col for col in twh_columns_to_drop if col in df.columns]
    twh = df.drop(twh_existing_columns).div(1e6)  # TWh
    twh = twh.groupby(level=1).sum()
    gw_columns_to_drop = ["Store", "Line"]
    gw_existing_columns = [col for col in gw_columns_to_drop if col in df.columns]
    gw = df.drop(gw_existing_columns).div(1e3)  # GW
    gen = gw.loc[["Generator", "StorageUnit"]].groupby(level=1).sum()
    con = gw.loc[["Link"]].groupby(level=1).sum()

    return gen, con, twh


def energy_balances(scenarios, sector, country):
    """
    Extracts and processes energy balance data for a specific sector and country
    from a collection of PyPSA-Eur network (.nc) files.

    This function supports aggregation of spatial and non-spatial results, adds derived carriers
    (e.g., oil refining from hydrogen inputs), and performs postprocessing for visualization.

    Parameters:
        scenarios (str): Path to the scenario folder that contains `configs/` and `networks/`.
        sector (str): The sector of interest (e.g., "energy", "Hydrogen_Storage", "co2").
        country (str): Country code (e.g., "DE") or "EU" for aggregated European results.

    Returns:
        pd.DataFrame: Energy balance for the specified sector in TWh, with scenarios as columns
                      and carriers as rows.
    """
    # Read config settings for scenario
    files = sorted([f for f in os.listdir(f"{scenarios}/configs") if os.path.isfile(os.path.join(f"{scenarios}/configs", f))])
    if files:
        with open(os.path.join(f"{scenarios}/configs", files[0]), 'r') as file:
            config = yaml.safe_load(file)

    # Get results from scenario (nc file)
    networks_dict = {
        (cluster, ll, opt + sector_opt, planning_horizon): f"{scenarios}"
        + f"/networks/base_s_{cluster}_{opt}_{sector_opt}_{planning_horizon}.nc"
        for cluster in config["scenario"]["clusters"]
        for opt in config["scenario"]["opts"]
        for sector_opt in config["scenario"]["sector_opts"]
        for ll in [config["electricity"]["transmission_limit"]]
        for planning_horizon in config["scenario"]["planning_horizons"]
    }

    # Set sceanrio settings in dataset
    columns = pd.MultiIndex.from_tuples(
        networks_dict.keys(),
        names=["cluster", "ll", "opt", "planning_horizon"],
    )
    energy_balance = pd.DataFrame(columns=columns, dtype=float)
    print(f"{sector} analysis for {country} and scenario {scenarios}")
    # Set results in dataset
    for label, filename in networks_dict.items():
        # Read nc file
        n = pypsa.Network(filename)
        # Extract and sort results from nc file
        if country == "EU":
            df_annual = n.statistics.energy_balance()
        else:           
            df_annual = n.statistics.energy_balance(groupby=["carrier", "bus_carrier", "country"])
            idx = pd.IndexSlice
            df_annual = df_annual.loc[idx[:,:,:,country]]
            
            # Add Non spatial EU supply
            # Get EU results structure and define missing country-specific carriers 
            df_annual_eu = n.statistics.energy_balance()
            carriers_to_add = ["oil", "coal", "lignite", "methanol", "oil primary", "uranium"]
            
            # Filter df_annual_eu for those bus_carriers
            df_filtered = df_annual_eu[df_annual_eu.index.get_level_values('bus_carrier').isin(carriers_to_add)]
            df_filtered[:] = 0.0

            # Merge placeholders for missing carriers (set to 0.0)
            df_annual = df_annual.add(df_filtered, fill_value=0)
            df_annual = df_annual.reorder_levels(['bus_carrier', 'component', 'carrier'])

            # Conversion for regional oil consideration
            df_annual.loc[("oil", "Link", "Fischer-Tropsch")] = df_annual.loc[("Hydrogen Storage", "Link", "Fischer-Tropsch")]*(-1)*0.8 if ("Hydrogen Storage", "Link", "Fischer-Tropsch") in df_annual.index else 0 
            df_annual.loc[("oil", "Link", "agriculture machinery oil")] = df_annual.loc[("agriculture machinery oil", "Link", "agriculture machinery oil")]*(-1) if ("agriculture machinery oil", "Link", "agriculture machinery oil") in df_annual.index else 0 
            df_annual.loc[("oil", "Link", "kerosene for aviation")] = df_annual.loc[("kerosene for aviation", "Link", "kerosene for aviation")]*(-1) if ("kerosene for aviation", "Link", "kerosene for aviation") in df_annual.index else 0 
            df_annual.loc[("oil", "Link", "land transport oil")] = df_annual.loc[("land transport oil", "Link", "land transport oil")]*(-1) if ("land transport oil", "Link", "land transport oil") in df_annual.index else 0 
            df_annual.loc[("oil", "Link", "oil")] = df_annual.loc[("naphtha for industry", "Link", "naphtha for industry")]*(-1) if ("naphtha for industry", "Link", "naphtha for industry") in df_annual.index else 0 
            df_annual.loc[("oil", "Link", "naphtha for industry")] = df_annual.loc[("AC", "Link", "oil")]*(-1)/0.35 if ("AC", "Link", "oil") in df_annual.index else 0
            df_annual.loc[("oil", "Link", "residential urban decentral oil boiler")] = df_annual.loc[("residential urban decentral heat", "Link", "residential urban decentral oil boiler")]*(-1)/0.79 if ("residential urban decentral heat", "Link", "residential urban decentral oil boiler") in df_annual.index else 0 
            df_annual.loc[("oil", "Link", "rural oil boiler")] = (df_annual.loc[("rural heat", "Link", "rural oil boiler")]*(-1)/0.82) if ("rural heat", "Link", "rural oil boiler") in df_annual.index else 0 
            df_annual.loc[("oil", "Link", "services rural oil boiler")] = df_annual.loc[("residential rural heat", "Link", "services rural oil boiler")]*(-1)/0.79 if ("residential rural heat", "Link", "services rural oil boiler") in df_annual.index else 0 
            df_annual.loc[("oil", "Link", "shipping oil")] = df_annual.loc[("shipping oil", "Link", "shipping oil")]*(-1) if ("shipping oil", "Link", "shipping oil") in df_annual.index else 0 
            df_annual.loc[("oil", "Link", "unsustainable bioliquids")] = df_annual.loc[("unsustainable bioliquids", "Link", "unsustainable bioliquids")]*(-1) if ("unsustainable bioliquids", "Link", "unsustainable bioliquids") in df_annual.index else 0 
            df_annual.loc[("oil", "Link", "urban decentral oil boiler")] = df_annual.loc[("urban decentral heat", "Link", "urban decentral oil boiler")]*(-1)/0.8 if ("urban decentral heat", "Link", "urban decentral oil boiler") in df_annual.index else 0 
            
            df_annual.loc[("oil", "Link", "oil refining")] = df_annual[df_annual.index.get_level_values('bus_carrier') == 'oil'].sum()*(-1) 
            
            df_annual.loc[("oil primary", "Generator", "oil primary")] = df_annual.loc[("oil", "Link", "oil refining")]/0.95 if ("oil", "Link", "oil refining") in df_annual.index else 0 
            df_annual.loc[("oil primary", "Link", "oil refining")] = df_annual.loc[("oil primary", "Generator", "oil primary")]*(-1) 
            
            # Conversion for regional coal consideration
            df_annual.loc[("coal", "Link", "coal")] = df_annual.loc[("AC", "Link", "coal")]*(-1)/0.33 if ("AC", "Link", "coal") in df_annual.index else 0 
            df_annual.loc[("coal", "Link", "coal for industry")] = df_annual.loc[("coal for industry", "Link", "coal for industry")]*(-1) if ("coal for industry", "Link", "coal for industry") in df_annual.index else 0 
            df_annual.loc[("coal", "Generator", "coal")] = (df_annual.loc[("coal", "Link", "coal")]+df_annual.loc[("coal", "Link", "coal for industry")])*(-1) 

            # Conversion for regional lignite consideration
            df_annual.loc[("lignite", "Link", "lignite")] = df_annual.loc[("AC", "Link", "lignite")]*(-1)/0.33 if ("AC", "Link", "lignite") in df_annual.index else 0 
            df_annual.loc[("lignite", "Generator", "lignite")] = df_annual.loc[("lignite", "Link", "lignite")]*(-1)

            # Conversion for regional methanol consideration
            df_annual.loc[("methanol", "Link", "CCGT methanol")] = df_annual.loc[("AC", "Link", "CCGT methanol")]*(-1)/0.57 if ("AC", "Link", "CCGT methanol") in df_annual.index else 0 
            df_annual.loc[("methanol", "Link", "CCGT methanol CC")] = df_annual.loc[("AC", "Link", "CCGT methanol CC")]*(-1)/0.57 if ("AC", "Link", "CCGT methanol CC") in df_annual.index else 0 
            df_annual.loc[("methanol", "Link", "Methanol steam reforming")] = df_annual.loc[("Hydrogen Storage", "Link", "Methanol steam reforming")]*(-1)/0.83 if ("Hydrogen Storage", "Link", "Methanol steam reforming") in df_annual.index else 0 
            df_annual.loc[("methanol", "Link", "Methanol steam reforming CC")] = df_annual.loc[("Hydrogen Storage", "Link", "Methanol steam reforming CC")]*(-1)/0.83 if ("Hydrogen Storage", "Link", "Methanol steam reforming CC") in df_annual.index else 0 
            df_annual.loc[("methanol", "Link", "OCGT methanol")] = df_annual.loc[("AC", "Link", "OCGT methanol")]*(-1)/0.41 if ("AC", "Link", "OCGT methanol") in df_annual.index else 0 
            df_annual.loc[("methanol", "Link", "industry methanol")] = df_annual.loc[("industry methanol", "Link", "industry methanol")]*(-1) if ("industry methanol", "Link", "industry methanol") in df_annual.index else 0 
            df_annual.loc[("methanol", "Link", "methanol-to-kerosene")] = df_annual.loc[("kerosene for aviation", "Link", "methanol-to-kerosene")]*(-1)*1.08 if ("kerosene for aviation", "Link", "methanol-to-kerosene") in df_annual.index else 0 
            df_annual.loc[("methanol", "Link", "methanolisation")] = df_annual.loc[("Hydrogen Storage", "Link", "methanolisation")]*(-1)/0.88 if ("Hydrogen Storage", "Link", "methanolisation") in df_annual.index else 0 
            df_annual.loc[("methanol", "Link", "shipping methanol")] = df_annual.loc[("shipping methanol", "Link", "shipping methanol")]*(-1) if ("shipping methanol", "Link", "shipping methanol") in df_annual.index else 0 
            
            # Conversion for regional nuclear consideration
            df_annual.loc[("uranium", "Link", "nuclear")] = df_annual.loc[("AC", "Link", "nuclear")]*(-1)/0.33 if ("AC", "Link", "nuclear") in df_annual.index else 0
        
        # Final Sorting with converted parameters
        df_annual = df_annual.reorder_levels(['bus_carrier', 'component', 'carrier'])
        df_annual = df_annual.sort_index()

        # Merge this file's results into the aggregated energy balance table
        energy_balance = energy_balance.reindex(df_annual.index.union(energy_balance.index))
        energy_balance.loc[df_annual.index, label] = df_annual
    # Fill nan values
    energy_balance = energy_balance.fillna(0)
    
    # Get sectors
    balances = {i.replace(" ", "_"): [i] for i in energy_balance.index.levels[0]}
    # Define sector energy
    co2_carriers = ["co2", "co2 stored", "process emissions"]
    balances["energy"] = [
        i for i in energy_balance.index.levels[0] if i not in co2_carriers
    ]

    for k, v in balances.items():
        if k == sector:
            # Filter results
            df = energy_balance.loc[energy_balance.index.get_level_values(0).isin(v)]
            df = df.groupby(df.index.get_level_values(2)).sum()
            # Convert MWh to TWh
            df = df / 1e6
            # Remove trailing link ports
            df.index = [
                (
                    i[:-1]
                    if (
                        (i not in ["co2", "NH3", "H2"])
                        and (i[-1:] in ["0", "1", "2", "3", "4"])
                    )
                    else i
                )
                for i in df.index
            ]
            # Drop non relevant scenario informations
            df = df.xs((config["scenario"]["clusters"][0], config["electricity"]["transmission_limit"], config["scenario"]["sector_opts"][0]), level=["cluster", "ll", "opt"], axis=1)

    return df 


def costs(scenarios, country):
    """
    Extracts and aggregates total system costs (CAPEX + OPEX) from PyPSA-Eur scenario outputs
    for a given country or for the EU as a whole.

    The function reads capital and marginal operating costs from all scenario combinations,
    filters for the relevant country (if specified), and returns a DataFrame with costs
    in billion Euros.

    Parameters:
        scenarios (str): Path to the scenario folder containing PyPSA networks and config files.
        country (str): Country code (e.g., "DE") or "EU" for full-European aggregation.

    Returns:
        pd.DataFrame: Multi-scenario cost table indexed by carrier and columns by planning horizon,
                      expressed in billion Euros (€/a).
    """
    # Read config settings for scenario
    files = sorted([f for f in os.listdir(f"{scenarios}/configs") if os.path.isfile(os.path.join(f"{scenarios}/configs", f))])
    if files:
        with open(os.path.join(f"{scenarios}/configs", files[0]), 'r') as file:
            config = yaml.safe_load(file)

    # Get results from scenario (nc file)
    networks_dict = {
        (cluster, ll, opt + sector_opt, planning_horizon): f"{scenarios}"
        + f"/networks/base_s_{cluster}_{opt}_{sector_opt}_{planning_horizon}.nc"
        for cluster in config["scenario"]["clusters"]
        for opt in config["scenario"]["opts"]
        for sector_opt in config["scenario"]["sector_opts"]
        for ll in [config["electricity"]["transmission_limit"]]
        for planning_horizon in config["scenario"]["planning_horizons"]
    }
    
    # Set sceanrio settings in dataset
    columns = pd.MultiIndex.from_tuples(
        networks_dict.keys(),
        names=["cluster", "ll", "opt", "planning_horizon"],
    )
    cost = pd.DataFrame(columns=columns, dtype=float)

    # Set results in dataset
    print(f"Cost analysis for {country} and scenario {scenarios}")
    for label, filename in networks_dict.items():
        # Read nc file
        n = pypsa.Network(filename)
        # Extract and sort results from nc file
        # Capital Expenditures (CAPEX)
        if country == "EU":
            df_capital_annual = pd.concat([n.statistics.capex()], keys=["capital"])
        else:
            df_capital_annual = pd.concat([n.statistics.capex(groupby=["carrier", "bus_carrier", "country"])], keys=["capital"])
            idx = pd.IndexSlice
            df_capital_annual = df_capital_annual.loc[idx[:,:,:,:,country]]
            df_capital_annual = df_capital_annual.reorder_levels([None, 'bus_carrier', 'component', 'carrier'])
            df_capital_annual = df_capital_annual.sort_index()

        cost = cost.reindex(df_capital_annual.index.union(cost.index))
        cost.loc[df_capital_annual.index, label] = df_capital_annual

        # Operating Expenditures (OPEX)
        if country == "EU":
            df_marginal_costs_annual = pd.concat([n.statistics.opex()], keys=["marginal"])
        else:
            df_marginal_costs_annual = pd.concat([n.statistics.opex(groupby=["carrier", "bus_carrier", "country"])], keys=["marginal"])
            idx = pd.IndexSlice
            df_marginal_costs_annual = df_marginal_costs_annual.loc[idx[:,:,:,:,country]]
            df_marginal_costs_annual = df_marginal_costs_annual.reorder_levels([None, 'bus_carrier', 'component', 'carrier'])
            df_marginal_costs_annual = df_marginal_costs_annual.sort_index()

        cost = cost.reindex(df_marginal_costs_annual.index.union(cost.index))
        cost.loc[df_marginal_costs_annual.index, label] = df_marginal_costs_annual
    # Fill nan values
    cost = cost.fillna(0)
    # Filter results
    if country == "EU":
        df = cost.groupby(cost.index.get_level_values(2)).sum()
    else:
        df = cost.groupby(cost.index.get_level_values(3)).sum()
    # Convert to billions
    df = df / 1e9
    # Drop non relevant scenario informations
    df = df.xs((config["scenario"]["clusters"][0], config["electricity"]["transmission_limit"], config["scenario"]["sector_opts"][0]), level=["cluster", "ll", "opt"], axis=1)
    
    return df


# Settings
# Set path for scenarios
MAIN_SCENARIOS = "/home/alex-charly/AP3/pypsa-eur/results/myopic/config.main"
LOWCARBON_SCENARIOS = "/home/alex-charly/AP3/pypsa-eur/results/myopic/config.lowcarbon"
LOWH2COST_SCENARIOS = "/home/alex-charly/AP3/pypsa-eur/results/myopic/config.lowH2cost"
GRIDFREEZE_SCENARIOS = "/home/alex-charly/AP3/pypsa-eur/results/myopic/config.gridfreeze"
HIGHH2DEMAND_SCENARIOS = "/home/alex-charly/AP3/pypsa-eur/results/myopic/config.highH2demand"
HIGHCARBON_SCENARIOS = "/home/alex-charly/AP3/pypsa-eur/results/myopic/config.highcarbon"

# Config settings for scenario
SCENARIOS = {
    (0, 0, 0, 0, 0): (MAIN_SCENARIOS), 
    (1, 0, 0, 0, 0): (LOWCARBON_SCENARIOS),
    (0, 1, 0, 0, 0): (LOWH2COST_SCENARIOS),
    (0, 0, 1, 0, 0): (GRIDFREEZE_SCENARIOS),
    (0, 0, 0, 1, 0): (HIGHH2DEMAND_SCENARIOS),
    (0, 0, 0, 0, 1): (HIGHCARBON_SCENARIOS),
}
NAMES = ["low_carbon", "low_h2cost", "grid_freeze", "high_h2demand", "high_carbon"]
countries = ["DE", "EU"] # EU means all European countries

for country in countries:
    # Cost preparation
    cost = pd.concat(
        {k: costs(scenarios, country) for k, (scenarios) in SCENARIOS.items()},
        names=NAMES,
    )
    cost.index.names = cost.index.names[:-1] + ["carrier"]
    cost = cost.stack([0]).to_xarray()
    cost.name = "costs"

    # Energy balance preparation
    energy = pd.concat(
        {k: energy_balances(scenarios, "energy", country) for k, (scenarios) in SCENARIOS.items()},
        names=NAMES,
    )
    energy.index.names = energy.index.names[:-1] + ["carrier"]
    energy = energy.stack([0]).to_xarray()
    energy.name = "energy"

    # CO2 balance preparation
    if country == "EU":
        co2 = pd.concat(
            {k: energy_balances(scenarios, "co2", country) for k, (scenarios) in SCENARIOS.items()},
            names=NAMES,
        )
        co2.index.names = co2.index.names[:-1] + ["carrier"]
        co2 = co2.stack([0]).to_xarray()
        co2.name = "co2"

    # H2 balance preparation
    h2 = pd.concat(
        {k: energy_balances(scenarios, "Hydrogen_Storage", country) for k, (scenarios) in SCENARIOS.items()},
        names=NAMES,
    )
    h2.index.names = h2.index.names[:-1] + ["carrier"]
    h2 = h2.stack([0]).to_xarray()
    h2.name = "hydrogen"
    
    # Generation, storage and conversion capacity preparation
    cap = pd.concat(
        {
            k: pd.concat(
                capacity(scenarios, country),
                keys=["generation", "conversion", "storage"],
            )
            for k, (scenarios) in SCENARIOS.items()
        },
        names=NAMES,
    )
    cap.index.names = cap.index.names[:-2] + ["category", "carrier"]
    cap = cap.stack([0]).unstack("category").to_xarray()
    
    # Merge and output all data
    if country == "EU":
        ds = xr.merge([cost, energy, co2, h2, cap]).round(2)
    else:
        ds = xr.merge([cost, energy, h2, cap]).round(2)
    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(f"{country}_scenarios_streamlit.nc", encoding=encoding)