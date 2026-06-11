# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A wind forecasting system for Squamish, BC. It scrapes weather data from Environment Canada and Squamish Windsports, stores it in SQLite, trains Temporal Fusion Transformer (TFT) models, and generates hourly (8-hour) and daily (5-day) forecasts with quality scores for windsports suitability.

## Running the System

### Core workflow

```bash
# 1. Download historical EC station CSVs + SWS wind data + initialise DB (first-time / refresh)
python collect_data.py
python collect_data.py --ec-only        # only EC station CSVs
python collect_data.py --sws-only       # only SWS wind data (requires Selenium + Chrome)
python collect_data.py --init-db        # only initialise SQLite DB

# 2. Build training datasets from downloaded raw data
python build_dataset.py                 # builds both hourly_database.csv and daily_database.csv
python build_dataset.py --mode hourly
python build_dataset.py --mode daily

# 3. Train models
python train.py --mode hourly           # 4 targets: speed, gust, lull, direction
python train.py --mode daily            # 4 targets: speed, hours_above_20, speed_score, direction_score
# Optional overrides: --epochs N --dropout F --hidden-size N --lr F --targets speed gust --no-lr-find
# Named experiment variants (saves to separate checkpoints, leaves production checkpoints intact):
#   --checkpoint-prefix tftExp    e.g. tftExpspeedHourlyCheckpoint.ckpt
#   --weight-boost 3.0            up-weight 10AM–6PM prediction timesteps by 3× during training

# 4. Update the SQLite DB with the latest EC + SWS observations (no forecast generated)
python update_data.py

# 5. Run forecasts (updates DB then generates CSV + JSON output)
python forecast.py                      # run both hourly and daily (one DB update)
python forecast.py --mode hourly
python forecast.py --mode daily
```

### Utility scripts

```bash
# Evaluate model accuracy against historical DB data
python evaluate.py --start 2025-06-20

# Compare two named model variants on the 2023 holdout across all 4 targets
python evaluate.py --compare            # expects tftBase* and tftWeighted* checkpoints
python evaluate.py --compare --windows 100

# Feature selection sweep
python feature_selection.py --mode hourly --epochs 2
python feature_selection.py --mode daily  --epochs 3
```

## Configuration

All model hyperparameters and feature lists live in `train_config.yaml` — it is the single source of truth. The Python dataclass defaults in `config.py` are fallbacks only; always edit the YAML. The `data.mask_intervals` section excludes known-bad sensor periods at training time without altering the CSV.

Notable YAML keys beyond features and sequence lengths: `sample_weight_boost` / `weight_target_start_hour` / `weight_target_end_hour` control time-of-day loss weighting during training; `val_full_data` / `val_predict_mode` control validation dataset construction (differ between hourly and daily).

Default checkpoint naming: `tft{target}{Hourly|Daily}Checkpoint.ckpt` (e.g., `tftspeedHourlyCheckpoint.ckpt`)
Named-variant naming (via `--checkpoint-prefix`): `{prefix}{target}{Hourly|Daily}Checkpoint.ckpt`
Dataset metadata naming: `{target}_training_dataset_{hourly|daily}.pkl` (prefixed variants: `{prefix}_{target}_...`)

## Data Flow

```
Environment Canada + Squamish Windsports
    ↓ (collect_data.py — historical; ec_scrape.py + sws_pull.py — live)
Per-station CSVs + sws_wind_database.csv + SQLite DB (weather_data_hourly.db)
    ↓ (build_dataset.py)
hourly_database.csv  daily_database.csv
    ↓ (train.py --mode hourly|daily)
TFT checkpoints (.ckpt) + dataset metadata (.pkl)
    ↓ (forecast.py)
hourly_speed_predictions.{csv,json}
daily_speed_predictions.{csv,json}
```

Training reads CSV snapshots. Inference reads live data from the SQLite DB + fresh EC forecast scrapes via `update_data.py`.

## Architecture

**Two parallel model pipelines — hourly and daily — each with 4 separate TFT models** (one per target variable). Quantile outputs (Q1–Q7) drive uncertainty-aware quality scores.

| | Hourly | Daily |
|---|---|---|
| Encoder length | 12 hours | 5 days |
| Prediction length | 8 hours | 5 days |
| Targets | speed, gust, lull, direction | speed, hours_above_20, speed_score, direction_score |
| Data source | `hourly_database.csv` | `daily_database.csv` |

**Feature selection finding:** Atmospheric pressures alone (Comox, Lillooet, Pam Rocks, Vancouver, Victoria) consistently outperform configs that add EC temperature forecasts at all epoch counts tested. EC temps are commented out of `train_config.yaml`; uncomment and re-run `feature_selection.py --mode hourly` to revisit.

**Known real features** (available in the forecast window): `sin_hour`, `year_fraction`.
**Unknown real features** (encoder past only): `comoxKPa`, `lillooetKPa`, `pamKPa`, `vancouverKPa`, `victoriaKPa`.

**Quality scores:**
- `speed_score` (1–5): steadiness; derived from gust/lull spread
- `direction_score` (1–5): directional consistency; derived from direction quantile spread
- `hours_above_20`: daily hours with speed > 20 knots
- `sailing_window`: boolean, speed > 15 knots

**Key modules:**
- `config.py` — `HourlyConfig` / `DailyConfig` dataclasses; populated from `train_config.yaml` via `from_yaml()`
- `tft_common.py` — shared `train_model()`, `_build_dataset()`, `_apply_mask_intervals()`, and `tft_with_ignore`; `_build_dataset` validates all feature columns exist before constructing the dataset
- `update_data.py` — `update_db()` pulls latest EC + SWS data into SQLite; `get_conditions_table_hourly/daily()` call it then build the inference DataFrame; runnable standalone for a DB-only update. Hourly path merges only DB history (no forecast scrape); daily path also merges `pull_forecast_daily()` for DegC real_known features.
- `forecast.py` — `_warn_missing()` fires before each fill step and prints NaN counts + max consecutive gap per feature; "ALL values missing" indicates a scraper or DB failure.
- `build_dataset.py` — builds training CSVs; also exports `add_scores_to_df()` used by `update_data.py` for daily inference
- `ec_scrape.py` — all live EC scraping: `pull_past_hrs_weather()`, `pull_forecast_daily()`; also exports `normalize_sky_series()`. (`pull_forecast_hourly()` exists but is no longer called — hourly model uses only pressure kPa from the DB encoder window, not forecast DegC/Sky.)
- `sws_pull.py` — Selenium-based SWS wind data fetch; `get_sws_df(dates)`

## Key Dependencies

- `pytorch-forecasting` — TFT model implementation
- `lightning.pytorch` — training loop
- `torch` — deep learning
- `pandas`, `numpy` — data wrangling
- `selenium` + Chrome — Squamish Windsports scraping
- `requests`, `beautifulsoup4` — Environment Canada scraping
- `thefuzz` — fuzzy HTML table parsing
- `python-dotenv` — `.env` loading
- `pytz` — Pacific timezone handling

All times are stored as Unix timestamps (INTEGER) in SQLite and handled in `America/Vancouver`.
