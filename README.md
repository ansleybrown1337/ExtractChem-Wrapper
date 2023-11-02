# ExtractChem Wrapper

A Python wrapper for the [ExtractChem equilibrium geochemical model](https://www.ars.usda.gov/pacific-west-area/riverside-ca/agricultural-water-efficiency-and-salinity-research-unit/docs/model/extractchem-model/).

## Author
Created by A.J. Brown  
Agricultural Data Scientist  
Email: ansleybrown1337@gmail.com

## Overview
This repository contains a set of Python scripts designed to facilitate the use of the ExtractChem model for geochemical equilibrium analysis. The code automates the generation of batch files from a CSV template with various soil characteristics and provides tools for graphing outputs post-simulation. It is particularly useful for conducting Monte Carlo analyses to evaluate the sensitivity of soil samples to ion adsorption, dissolution, and precipitation during changes in moisture content, and the subsequent effects on soil solution electrical conductivity.

## About the ExtractChem Model

**Year**: 2007  
**Version**: 2.0  

**Description**:  
The ExtractChem model, developed by the USDA Agricultural Research Service, predicts the major ion, boron composition, electrical conductivity (EC), and osmotic pressure (OP) of a soil solution at a desired water content based on the known ion composition at another water content.

**Major Uses**:  
- Predicting saturation extract composition, EC, and OP from analysis of soil water extracts (1:1, 1:2, or 1:5).
- Correcting soil water extract data for gypsiferous soils during calibration of field electromagnetic or electrical resistivity surveys for salinity.
- Calculating soil water concentrations for UNSATCHEM or other chemical transport models based on soil extract analyses.

**Basis of the Model**:  
ExtractChem is based on the chemical routines from the UNSATCHEM model, accounting for major ions, boron, and processes such as cation exchange, precipitation-dissolution of calcite and gypsum, and boron adsorption using the constant capacitance model.

## Workflow

To use this wrapper effectively, follow the general workflow:

1. **Batch File Creation**: Execute `ExtractChem_batch_merge.py` to create a batch file from your CSV template.
2. **Model Execution**: Manually open ExtractChem and run the batch file generated in the previous step. (Note: ExtractChem cannot be run via the terminal.)
3. **Result Graphing**: Run `EC_graphs.py` to collect output data and display the results. This script depends on the `GOF_stats.py` module, which you can find in the `Code` folder.

## Getting Started

> !NOTE
> I haven't had time to make much help documentation on this yet; please use the **Example Run in Ipython.txt** file for guidance. I'm also happy to help if you reach out to me via email for specific applications and tutorials.

## References
- D.L. Suarez and J. Simunek, 1997. UNSATCHEM: Unsaturated water and solute transport model with equilibrium and kinetic chemistry. Soil Sci. Soc. Am. J. 61:1633-1646.
- Suarez, D.L., and J. Simunek. 1996. UNSATCHEM code for simulating one-dimensional variably saturated water flow, heat transport, carbon dioxide production and transport, and multicomponent solute transport with major ion equilibrium and kinetic chemistry. U.S. Salinity Laboratory Tech. Report No. 129.

Python "wrapper for the ExtractChem equilbrium geochemical model."

Created by A.J. Brown


I haven't had time to make much help documentation on this yet, so I'm happy to help if you reach out to me for specific applications and tutorials.

Cheers,
AJ