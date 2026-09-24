# A Decade of PhotoMappers: A Longitudinal Study of Crowdsourced Disaster Photos Geolocalization with Vision-Language Models and Geospatial Reasoning

### Project Overview

This repository contains a reproducible notebook-first workflow for preparing, analyzing, and visualizing PhotoMapper geolocation and reasoning data. The pipeline covers data cleaning, GeoJSON conversion, state-level spatial aggregation, longitudinal indicator analysis, lifeline analysis, and publication-ready visual summaries.

## Project Highlights

- End-to-end notebook workflow with consistent file paths.
- Clean root layout with centralized data under one folder.
- Reproducible exports for both tabular and figure outputs.
- Shared helper utilities in a reusable Python module.

## Repository Layout

```text
github_release/
├── 01_data_record_analysis.ipynb
├── 02_excel_to_geojson.ipynb
├── 03_ Choropleth_Map.ipynb
├── 04_Longitudinal_Reasoning_Analysis.ipynb
├── 05_Longitudinal_lifelines.ipynb
├── 06_PhotoMappers_Geolocation_Visualization.ipynb
├── photomapper_common.py
└── data/
    ├── input/
    ├── shapefiles/
    ├── geojson_output/
    ├── outputs/
    └── raw-record/
```

## Environment Setup

Use Python 3.10+ (tested on Python 3.13 in VS Code/Jupyter).

```bash
conda create -n photomapper python=3.13 -y
conda activate photomapper
pip install pandas numpy matplotlib seaborn geopandas shapely pyproj fiona mapclassify openpyxl jupyter
```

If you already work inside an existing environment, install only missing packages.

## Data Organization

Place inputs in:

- `data/input/`
  - `dataset_12663_with_disaster_type_refined.xlsx`
  - `update_dataset_with_disaster_type20260917.xlsx`
  - `reasoning_yearly_2017_2025_20260918.xlsx`
  - `reasoning_lifeline_20260918.xlsx`

Place U.S. state shapefiles in:

- `data/shapefiles/`
  - `cb_2025_us_state_5m.shp` and associated sidecar files (`.dbf`, `.shx`, `.prj`, etc.)

Generated products are written to:

- `data/geojson_output/`
- `data/outputs/`

## Workflow (Run Order)

Run notebooks in the following order:

1. `01_data_record_analysis.ipynb`
2. `02_excel_to_geojson.ipynb`
3. `03_ Choropleth_Map.ipynb`
4. `04_Longitudinal_Reasoning_Analysis.ipynb`
5. `05_Longitudinal_lifelines.ipynb`
6. `06_PhotoMappers_Geolocation_Visualization.ipynb`

This order ensures each downstream notebook can consume previously generated outputs.

## Notebook Summary

### 1) Data Record Analysis

- Loads refined workbook input.
- Standardizes and summarizes record-level attributes.
- Exports cleaned workbook and descriptive distributions.

### 2) Excel to GeoJSON

- Validates coordinates and key fields.
- Converts records to GeoJSON.
- Produces corrected/filtered geospatial files.

### 3) State Choropleth Analysis

- Joins points with state polygons.
- Aggregates yearly and total counts.
- Exports state-level GeoJSON/CSV artifacts and map figures.

### 4) Longitudinal Reasoning Analysis

- Parses indicator-level yearly worksheets.
- Computes overall, cue-level, and evidence-level trends.
- Exports summary tables and longitudinal figures.

### 5) Lifeline Reasoning Analysis

- Parses FEMA lifeline indicator sheets.
- Handles missingness and zero values explicitly.
- Exports lifeline metrics and heatmap figures.

### 6) Geolocation Visualization

- Consumes outputs from previous analyses.
- Builds publication-style multi-panel geolocation figures.

## Reproducibility Notes

- Notebooks use project-relative paths from repository root.
- Use the shared helper module `photomapper_common.py` for common parsing logic.
- If a notebook kernel is missing packages, install them in the active environment and rerun cells in order.

## Citation

If you use this workflow in academic work, please cite:

```bibtex
@misc{li2026photomappers,
  title = {A Decade of PhotoMappers: A Longitudinal Study of Crowdsourced Disaster Photos Geolocalization with Vision-Language Models and Geospatial Reasoning},
  author = {Li, Hao and Yin, Wenping and Deuser, Fabian and Jia, Jia and Liu, Ziqi and Juhasz, Levente and Zhang, Fan and Biljecki, Filip},
  year = {2026},
  note = {Corresponding author: Hao Li (hao.li@nus.edu.sg)}
}
```

## Acknowledgment

This repository design is inspired by research-oriented open-source releases that combine clear dataset instructions, reproducible workflows, and publication-ready outputs.
