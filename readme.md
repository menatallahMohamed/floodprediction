# Flood Prevention Using River Level Prediction

LSTM and Attention-LSTM models that predict the next-minute river level from the previous 60 minutes of multivariate weather, rainfall, and river level sensor data in Dublin, Ireland (2021–2025).
The data is provided by Smart Dublin https://data.smartdublin.ie/

## Overview

This project investigates whether the upcoming minute's river level can be accurately predicted from the preceding 60-minute window of weather, rainfall, and river level observations, enabling early flood warnings. The pipeline covers:

1. **Data merging** — consolidating monthly CSV exports from Dublin City Council sensors into unified parquet files.
2. **Proximity mapping** — matching weather stations, rainfall gauges, and river level monitors by geographic proximity.
3. **Preprocessing** — leakage-safe domain-rule cleaning, feature engineering, and versioned dataset exports.
4. **Exploratory data analysis** — distribution, correlation, stationarity, and missing-data analysis.
5. **Modelling** — stacked LSTM, 3-layer LSTM, Attention-LSTM, and feature ablation experiments with naive and linear baselines.
6. **Comparison & reporting** — cross-model metric comparison and LaTeX table generation.

## Dataset

The raw data comes from Dublin City Council's public sensor network and covers **2021-01 to 2025-12**. After merging and cleaning, the final modelling dataset contains **10,678,050 rows** across **4 locations**:

| Location | River Level Monitor | Monitor Type |
|---|---|---|
| Ballymun Coultry Parks Depot | Beaumont Road (The Wad) | Culverted River Monitor |
| Civic Offices | Rory O'Moore Bridge | Tide Monitor |
| Kippure | Bohernabreena Weir (Dodder River) | River Monitor |
| Pigeon House | Pigeon House | Tide Monitor |

**Features (8 after encoding):** windspeed, humidity, temperature, dewpoint, pressure, rainfall, wind direction sin, wind direction cos

**Target:** river level (metres)

> **Note:** Bedford Lane was excluded due to ~60% missing rainfall and a sensor plateau artefact. Ballymun is present in the dataset but excluded from LSTM batch runs due to weak weather–river correlations at its culverted monitor.

### Data availability

Raw CSV files are not included in this repository due to size. They can be obtained from the [Dublin City Council Smart Dublin](https://data.smartdublin.ie/) open data portal. Place them under `datasets/` following this structure:

```
datasets/
├── weather/2021/  ... 2025/   # Monthly weather CSVs
├── rainfall/2021/ ... 2025/   # Monthly rainfall CSVs
└── riverlevel/2021/ ... 2025/ # Monthly river level CSVs
```

## Requirements

- Python 3.12+
- Key dependencies:

| Package | Purpose |
|---|---|
| tensorflow | LSTM / Attention-LSTM training |
| scikit-learn | Scaling, linear baseline, metrics |
| pandas | Data manipulation |
| numpy | Numerical computation |
| pyarrow | Parquet I/O |
| matplotlib | Plotting |
| statsmodels | Stationarity tests (ADF, KPSS) |
| scipy | Statistical analysis |

Install all dependencies:

```bash
pip install tensorflow pandas numpy pyarrow scikit-learn matplotlib statsmodels scipy
```

## Pipeline Reproduction

Run from the project root with the virtual environment activated:

```bash
# 1. Merge raw CSVs into per-source parquet files
python merge_20212025_to_parquet.py weather
python merge_20212025_to_parquet.py rainfall
python merge_20212025_to_parquet.py riverlevel

# 2. Join weather + rainfall
python join_weather_rainfall_20212025_v2.py

# 3. Join weather+rainfall + river level
python join_wr_riverlevel.py

# 4. Create leakage-safe cleansed version
python preprocess_weather_rainfall_riverlevel.py --version v20260309_02

# 5. Run EDA (optional)
python eda_weather_rainfall_riverlevel.py

# 6. Train models
python lstm_60stepsback_batch.py --location "Civic Offices" "Kippure" "Pigeon House"
python lstm_60stepsback_3layer_batch.py --location "Civic Offices" "Kippure" "Pigeon House"
python lstm_60stepsback_ablation_batch.py --location "Kippure" --year 2021
python attention_lstm_river_level.py --run-all-observed
```

## Project Structure

```
├── SCRIPTS.md                          # Detailed script documentation
├── README.md
│
├── # ── Data Pipeline ──
├── merge_20212025_to_parquet.py        # Merge monthly CSVs → parquet
├── merge_rainfall_20212025_to_parquet.py
├── join_weather_rainfall_20212025_v2.py # Join weather + rainfall
├── join_wr_riverlevel.py               # Join with river level
├── preprocess_weather_rainfall_riverlevel.py  # Leakage-safe cleaning
│
├── # ── Exploratory Data Analysis ──
├── eda_weather_rainfall_riverlevel.py
├── stationarity_tests.py              # ADF & KPSS stationarity tests
├── weather_rainfall_riverlevel_eda.ipynb
├── CategoricalAnalysis.ipynb
│
├── # ── Modelling ──
├── lstm_60stepsback_attention_3layer_uniformscaling_batch.py   # 3-layer LSTM variant
├── lstm_60stepsback_attention_uniformscaling_batch.py   # 2-layer LSTM (core)
       
├── lstm_60stepsback_3layer_batch.py   
├── lstm_60stepsback_uniformscaling_batch.py
├── lstm_120stepsback_uniformscaling_batch.py

├── lstm_60stepsback_ablation_uniformscaling_grouped_batch.py # Feature ablation study

├── attention_lstm_river_level.py       # Attention-LSTM with explainability
│
├── # ── Analysis & Reporting ──
├── compare_lstm_vs_attention.py
├── regenerate_ablation_plots.py

│
├── # ── Notebooks ──
├── LSTM-60stepsback.ipynb
├── LSTM-modified.ipynb
├── LSTM-revisited.ipynb
├── MultipleRegressions-perlocation.ipynb
├── mergefiles20212025.ipynb
│
├── ProximityMapping/                   # Station mapping reference data
└── analysis_outputs/                   # Generated outputs (not tracked)
```

## Modelling Approach

All LSTM scripts share these conventions:

- **Per-location, per-year training** — each location–year combination is an independent experiment.
- **Chronological splits** — 80/20 train/test (no shuffling); the attention model uses 72/8/20 train/val/test.
- **Training-only imputation & scaling** — feature means and scalers are fitted on the training split only to prevent data leakage.
- **Gap-aware sequences** — 60-step sliding windows that break at time gaps > 5 minutes.
- **Wind direction encoding** — raw degrees → sin/cos components.
- **Baselines** — every run includes naive (last-value) and linear regression baselines.

### Model Architectures

| Model | Layers | Notes |
|---|---|---|
| 2-Layer LSTM | LSTM(64) → LSTM(32) → Dense(16) → Dense(1) | Core model, Huber loss |
| 3-Layer LSTM | LSTM(64) → LSTM(32) → LSTM(16) → Dense(16) → Dense(1) | Deeper variant |
| Attention-LSTM | LSTM(64) → LSTM(32) → Temporal Attention → Dense(16) → Dense(1) | With early stopping, LR scheduling, and explainability outputs |

### Explainability (Attention-LSTM)

The attention model produces:
- **Temporal attention weights** — which lag steps the model attends to.
- **Permutation feature importance** — RMSE degradation when each feature is shuffled.
- **Integrated gradients** — per-timestep, per-feature attribution maps.
- **Local explanations** — top contributing features for individual predictions.

