# Burn Scar Analysis from Sentinel-2 Imagery

This project implements a baseline model for detecting burn scars from satellite imagery using the differenced Normalized Burn Ratio (dNBR).

## 1. Description

The script ingests pre-fire and post-fire Sentinel-2 bands (specifically NIR and SWIR), calculates the dNBR, and applies a threshold to generate a binary burn mask. This serves as the baseline for a more advanced machine learning model.

## 2. Setup

First, create and activate a virtual environment. This project uses `uv` for package management.

```bash
uv venv
source .venv/bin/activate
```

Next, install the required dependencies from the `pyproject.toml` file:

```bash
uv pip install -e .
```

## 3. Data

Download the required Sentinel-2 bands and place them in a `bands/` directory in the root of the project. This directory is included in the `.gitignore` and should not be committed to the repository.

The required bands are:

- **Near-Infrared (NIR):** Band 8A
- **Short-Wave Infrared (SWIR):** Band 12

The expected file structure is:

```
.
├── bands/
│   ├── pre_B8A.jp2
│   ├── pre_B12.jp2
│   ├── post_B8A.jp2
│   └── post_B12.jp2
└── ... (other project files)
```

## 4. Usage

To run the analysis and generate the burn mask plot, execute the main script:

```bash
uv run main.py
```
