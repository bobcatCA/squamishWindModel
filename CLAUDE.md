# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A wind forecasting system for Squamish, BC. It scrapes weather data from Environment Canada and Squamish Windsports, stores it in SQLite, trains Temporal Fusion Transformer (TFT) models, and generates hourly (8-hour) and daily (5-day) forecasts with quality scores for windsports suitability.

## Running the System

```bash
# Train hourly models (4 targets: speed, gust, lull, direction)
python train.py --mode hourly

# Train daily models (4 targets: speed, hours_above_20, speed_score, direction_score)
python train.py --mode daily

# Optional overrides: --epochs N --dropout F --hidden-size N --lr F --targets speed gust --no-lr-find
# Named experiment variants (saves to separate checkpoints, leaves production checkpoints intact):
#   --checkpoint-prefix tftExp    e.g. tftExpspeedHourlyCheckpoint.ckpt
#   --weight-boost 3.0            up-weight 10AM–6PM prediction timesteps by 3× during training

# Update the SQLite DB with the latest EC + SWS observations (no forecast generated)
python updateWeatherData.py

# Run hourly forecast (updates DB then generates CSV + JSON output)
python forecast_hourly.py

# Run daily forecast (updates DB then generates CSV + JSON output)
python forecast_daily.py

# Evaluate hourly model against historical data (from June 20, 2025)
python evaluate_hourly.py

# Compare two named model variants on the 2023 holdout across all targets
python compare_models.py          # expects tftBase* and tftWeighted* checkpoints

# Rebuild training CSVs from EC station CSVs + SWS wind data
python build_dataset.py           # → hourly_database.csv
python score_daily.py             # → daily_database.csv  (reads hourly_database.csv)

# Feature selection experiments
python feature_selection.py --epochs N        # hourly
python feature_selection_daily.py --epochs N  # daily

# Data collection
python ec_history.py              # Download historical EC station CSVs (Jan 2016 → present)
python sws_pull.py                # Download historical SWS wind data (requires Selenium/Chrome)
# Live data (called automatically by forecast scripts via updateWeatherData.py):
#   ec_scrape.py — pull_past_hrs_weather(), pull_forecast_hourly(), pull_forecast_daily()
#   sws_pull.py  — get_sws_df()

# Database
python db_init.py                 # Initialise SQLite schema (run once)
```

## Configuration

A `.env` file must define `WORKING_DIRECTORY` — this is where the SQLite database (`weather_data_hourly.db`), trained checkpoints (`.ckpt` files), training dataset metadata (`.pkl` files), and output forecasts are stored.

All model hyperparameters and feature lists live in `train_config.yaml` — it is the single source of truth. The Python dataclass defaults in `config.py` are fallbacks only; always edit the YAML. The `data.mask_intervals` section excludes known-bad sensor periods at training time without altering the CSV.

Notable YAML keys beyond features and sequence lengths: `sample_weight_boost` / `weight_target_start_hour` / `weight_target_end_hour` control time-of-day loss weighting during training; `val_full_data` / `val_predict_mode` control validation dataset construction (differ between hourly and daily).

Default checkpoint naming: `tft{target}{Hourly|Daily}Checkpoint.ckpt` (e.g., `tftspeedHourlyCheckpoint.ckpt`)
Named-variant naming (via `--checkpoint-prefix`): `{prefix}{target}{Hourly|Daily}Checkpoint.ckpt`
Dataset metadata naming: `{target}_training_dataset_{hourly|daily}.pkl` (prefixed variants: `{prefix}_{target}_...`)

## Data Flow

```
Environment Canada + Squamish Windsports
    ↓ (ec_scrape.py, sws_pull.py — live; ec_history.py — bulk historical)
SQLite DB (weather_data_hourly.db)
    ↓ (build_dataset.py)         ← also reads EC station CSVs directly
hourly_database.csv  daily_database.csv
    ↓ (train.py --mode hourly|daily)
TFT checkpoints (.ckpt) + dataset metadata (.pkl)
    ↓ (forecast_hourly.py / forecast_daily.py)
hourly_speed_predictions.{csv,json}
daily_speed_predictions.{csv,json}
```

Training reads CSV snapshots. Inference reads live data from the SQLite DB + fresh EC forecast scrapes via `updateWeatherData.py`.

## Architecture

**Two parallel model pipelines — hourly and daily — each with 4 separate TFT models** (one per target variable). Quantile outputs (Q1–Q7) drive uncertainty-aware quality scores.

| | Hourly | Daily |
|---|---|---|
| Encoder length | 12 hours | 5 days |
| Prediction length | 8 hours | 5 days |
| Targets | speed, gust, lull, direction | speed, hours_above_20, speed_score, direction_score |
| Data source | `hourly_database.csv` | `daily_database.csv` |

**Feature selection finding:** Atmospheric pressures alone (Comox, Lillooet, Pam Rocks, Vancouver, Victoria) consistently outperform configs that add EC temperature forecasts at all epoch counts tested. EC temps are commented out of `train_config.yaml`; uncomment and re-run `feature_selection.py` to revisit.

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
- `updateWeatherData.py` — `update_db()` pulls latest EC + SWS data into SQLite; `get_conditions_table_hourly/daily()` call it then build the inference DataFrame; runnable standalone (`python updateWeatherData.py`) for a DB-only update with no forecast
- `score_daily.py` — computes daily target labels (`add_scores_to_df()`); also runnable standalone to rebuild `daily_database.csv`
- `compare_models.py` — evaluates two named model variants (default: `tftBase*` vs `tftWeighted*`) against the 2023 holdout across all targets; prints per-target MAE tables
- `db_init.py` — SQLite schema (run once to initialise)
- `ec_scrape.py` — all live EC scraping: `pull_past_hrs_weather()`, `pull_forecast_hourly()`, `pull_forecast_daily()`; also exports `normalize_sky_series()` used by both `build_dataset.py` and `updateWeatherData.py`
- `ec_history.py` — bulk historical download from EC climate API → per-station CSVs
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
