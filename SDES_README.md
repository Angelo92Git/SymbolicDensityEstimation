# READ ME

## branch guide
 - master: used to sync fork with SymbolicRegression.jl 
 - dev: main development branch

## commit guide
 - feat: new features
 - fix: bug fixes
 - docs: documentation updates
 - config: configuration files
 - data: data files
 - script: bash scripts
 - del: deleted files

 ## Overview
 This repository contains code for Symbolic Density Estimation (SDE) using Symbolic Regression techniques
 The main pipeline is implemented in `sdes_pipeline.jl`, which orchestrates data processing, symbolic regression, and result saving.
 The code leverages the `SymbolicRegression.jl` package for performing symbolic regression tasks.
 The repository also includes configuration files for different experimental setups and datasets in the `config_management` directory.
 The `data` directory contains raw and processed datasets used in the experiments.
 The `logs` directory is used to store logs and results from the symbolic regression runs.
 The `data_exploration` directory contains scripts and notebooks for exploring and visualizing the datasets.
 The script `gen_data.py` is used to produce the KDE estimates for the datasets which is then used in the `sdes_pipeline.jl` script.