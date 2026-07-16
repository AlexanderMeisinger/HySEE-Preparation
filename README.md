# HySEE-Preparation
HySEE - Hydrogen in Southeast Europe

This repository contains the 

## Project description
HySEE identifies suitable areas for renewable hydrogen (H2) production in Bulgaria and Romania and support the establishment of a South-Eastern H2 Corridor that can supply Central Europe, including Germany. To achieve this, the project map renewable energy potential, production capacity, and current and future hydrogen demand.

## Installation and usage
1. Open your terminal at a location where you want to install the repository HySEE including it's subworkflows PyPSA-Earth. Type the following in your terminal to download the packages and the dependencies (pypsa-earth) from GitHub. Note that the tag `--recursive-submodules` is needed to automatically clone also the pypsa-eur and pypsa-earth dependencies.
   
   ```bash
   git clone --recurse-submodules https://github.com/AlexanderMeisinger/HySEE-Preparation.git
   ```
   
3. Move the current directory to the head of the repository.
   
   ```bash
   .../some/path/without/spaces % cd HySEE-Preparation
   ```
   
4. The PyPSA-Earth python package requirements are curated in the `workflow/pypsa-earth/envs/environment.yaml` of the PyPSA-Earth respository. The environment can be installed using conda or mamba:
   
   ```bash
   .../HySEE-Preparation % conda env create -f workflow/pypsa-earth/envs/environment.yaml
   ```
   
5. For running the optimisation one has to install the solver. We can recommend the open source `HiGHs` solver, see more details on solvers in the documentation of [PyPSA-Earth](https://pypsa-earth.readthedocs.io/en/latest/index.html). The optimisation in this work was performed using the commercial `Gurobi` solver.

## Run scenarios
For starting the PyPSA-Earth model, run the following command:
```bash
.../HySEE-Preparation/workflow/pypsa-earth % conda activate pypsa-earth
```
```bash
% Quick check
% snakemake --cores all solve_sector_networks --configfile .../HySEE-Preparation/config/country/config.yaml -n
```
```bash
% Full run
% snakemake --cores all solve_sector_networks --configfile .../HySEE-Preparation/config/country/config.yaml
```

Please follow the documentation of [PyPSA-Earth](https://pypsa-earth.readthedocs.io/en/latest/index.html) for more details. The estimated time to run one single optimisation is 120 mins on a standard laptop. To run the full set, a high-performance computer is recommended. The models for Bulgaria and Romania are executed independently.

## Reproducibility
The project results and analysis can be reproduced using the notebooks in `workflow/notebooks` after successfully running the scenarios in `config`.

## Result and input data
A dataset of the model results will be available on Zenodo under a CC-BY-4.0 license. Please refer to the documentation of [PyPSA-Earth](https://pypsa-earth.readthedocs.io/en/latest/index.html) for details on the input data.

## License
The code in this repo is MIT licensed, see ./LICENSE.md.