# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A wind forecasting system for Squamish, BC. It scrapes weather data from Environment Canada and Squamish Windsports, stores it in SQLite, trains Temporal Fusion Transformer (TFT) models, and generates hourly (8-hour) and daily (5-day) forecasts with quality scores for windsports suitability.

## Running the System

### Core workflow

```bash
# 1. Download historical EC station CSVs + SWS wind data + initialise DB (first-time / refresh)
python collect_data.py
python collect_data.py --ec-only        # only EC station CSVs (full rebuild → web_data/)
python collect_data.py --ec-update      # append EC data since last CSV entry (detects/re-fetches sparse months)
python collect_data.py --sws-only       # only SWS wind data (requires Selenium + Chrome)
python collect_data.py --sws-update     # append SWS data since last CSV entry
python collect_data.py --init-db        # only initialise SQLite DB

# 2. Build training datasets from downloaded raw data
python build_dataset.py                 # builds both training_data/hourly_database.csv and daily_database.csv
python build_dataset.py --mode hourly
python build_dataset.py --mode daily

# 3. Train models
python train.py --mode hourly           # targets defined in train_config.yaml (e.g. squamishSpeed)
python train.py --mode daily            # targets defined in train_config.yaml
# Optional overrides: --epochs N --dropout F --hidden-size N --lr F --targets speed gust --no-lr-find
# Named experiment variants (saves to separate checkpoints, leaves production checkpoints intact):
#   --checkpoint-prefix tftExp    e.g. tftExpspeedHourlyCheckpoint.ckpt

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

# Plot predicted vs actual at multiple forecast horizons (1h/4h hourly; 1d daily)
python horizon_eval.py                        # hourly, all targets, stride=4h
python horizon_eval.py --mode daily           # daily, all targets, stride=1d
python horizon_eval.py --target speed         # single target
python horizon_eval.py --start 2025-06-01 --save

# Feature selection sweep
python feature_selection.py --mode hourly --epochs 2
python feature_selection.py --mode daily  --epochs 3
```

## Configuration

All model hyperparameters and feature lists live in `train_config.yaml` — it is the single source of truth. The Python dataclass defaults in `config.py` are fallbacks only; always edit the YAML. The `data.mask_intervals` section excludes known-bad sensor periods at training time without altering the CSV.

Notable YAML keys beyond features and sequence lengths: `val_full_data` / `val_predict_mode` control validation dataset construction (differ between hourly and daily).

Default checkpoint naming: `models/tft{target}{Hourly|Daily}Checkpoint.ckpt` (e.g., `models/tftspeedHourlyCheckpoint.ckpt`)
Named-variant naming (via `--checkpoint-prefix`): `models/{prefix}{target}{Hourly|Daily}Checkpoint.ckpt`
Dataset metadata naming: `models/{target}_training_dataset_{hourly|daily}.pkl` (prefixed variants: `models/{prefix}_{target}_...`)

## Data Flow

```
Environment Canada + Squamish Windsports
    ↓ (collect_data.py — historical; ec_scrape.py + sws_pull.py — live)
web_data/{station}.csv + web_data/sws_wind_database.csv + web_data/weather_data_hourly.db
    ↓ (build_dataset.py)
training_data/hourly_database.csv  training_data/daily_database.csv
    ↓ (train.py --mode hourly|daily)
models/tft*Checkpoint.ckpt + models/*_training_dataset_*.pkl
    ↓ (forecast.py)
forecasts/hourly_speed_predictions.{csv,json}
forecasts/daily_speed_predictions.{csv,json}
```

Training reads CSV snapshots. Inference reads live data from the SQLite DB + fresh EC forecast scrapes via `update_data.py`.

## Architecture

**Two parallel model pipelines — hourly and daily — each with one TFT model per target variable** (targets configured in `train_config.yaml`). Quantile outputs (Q1–Q7) drive uncertainty-aware quality scores.

| | Hourly | Daily |
|---|---|---|
| Encoder length | 12 hours | 5 days |
| Prediction length | 8 hours | 5 days |
| Targets (current) | squamishSpeed | squamishSpeed |
| Data source | `training_data/hourly_database.csv` | `training_data/daily_database.csv` |

**Column naming convention:**
- `squamishSpeed` / `squamishGust` / `squamishLull` / `squamishDirection` / `squamishDegC` — SWS sensor at the Squamish spit
- `{station}DegC` / `{station}KPa` / `{station}Hum` / `{station}Sky` — EC weather stations

The SQLite DB stores SWS columns under legacy names (`speed`, `gust`, `lull`, `direction`, `temperature`). `update_data.py` applies `_DB_COL_RENAME` when reading from the DB so all downstream code uses the `squamish*` convention.

**Feature selection finding (earlier):** Atmospheric pressures alone (Comox, Lillooet, Pam Rocks, Vancouver, Victoria) outperformed configs adding EC temperature forecasts at the epoch counts tested. Currently under re-evaluation — the active `train_config.yaml` uses EC temperatures (DegC) as `real_unknown` for hourly, not pressures. Re-run `feature_selection.py --mode hourly` to compare.

**Known real features** (available in the forecast window): `year_fraction` (see YAML for which are active).
**Unknown real features** (encoder past only): pressure kPa and/or station DegC columns — see `real_unknown` in `train_config.yaml` for current active set.

**Quality scores:**
- `speed_score` (1–5): steadiness; derived from gust/lull spread
- `direction_score` (1–5): directional consistency; derived from direction quantile spread
- `hours_above_20`: daily hours with speed > 20 knots
- `sailing_window`: boolean, speed > 15 knots

**Key modules:**
- `config.py` — `HourlyConfig` / `DailyConfig` dataclasses; populated from `train_config.yaml` via `from_yaml()`
- `tft_common.py` — shared `train_model()`, `_build_dataset()`, `_apply_mask_intervals()`, and `tft_with_ignore`; `_build_dataset` validates all feature columns exist and omits empty feature lists (YAML parses `[]` as `None`)
- `update_data.py` — `update_db()` pulls latest EC + SWS data into SQLite; `get_conditions_table_hourly/daily()` call it then build the inference DataFrame; applies `_DB_COL_RENAME` to map legacy DB column names to `squamish*`; runnable standalone for a DB-only update. Hourly path merges only DB history (no forecast scrape); daily path also merges `pull_forecast_daily()` for DegC real_known features.
- `forecast.py` — `_warn_missing()` fires before each fill step and prints NaN counts + max consecutive gap per feature; "ALL values missing" indicates a scraper or DB failure.
- `build_dataset.py` — builds training CSVs; also exports `add_scores_to_df()` used by `update_data.py` for daily inference
- `ec_scrape.py` — all live EC scraping: `pull_past_hrs_weather()`, `pull_forecast_daily()`; also exports `normalize_sky_series()`. (`pull_forecast_hourly()` exists but is no longer called — hourly inference uses only DB encoder window data, not forecast DegC/Sky.)
- `horizon_eval.py` — plots predicted vs actual at multiple horizons from the live DB; supports `--mode hourly` (1h/4h) and `--mode daily` (1d with Q25–Q75 band)
- `sws_pull.py` — Selenium-based SWS wind data fetch; `get_sws_df(dates)` for arbitrary date lists; `update_sws_csv()` for incremental append since last CSV entry

## Plain NN Baseline (simple_model.py)

A same-timestep regression baseline — no encoder/decoder windows, no quantile loss — built to isolate whether the configured features explain `squamishSpeed` at all, independent of TFT's sequence-modeling machinery.

```bash
python simple_model.py --train                   # train + save models/simple_nn_hourly.pt
python simple_model.py --test [--source db|csv]   # predict + plot; csv exposes Hum features the live DB lacks
python simple_model.py                            # train then test
```

Reads the same `hourly:` `real_known` / `real_unknown` / `categorical` lists from `train_config.yaml` as the TFT pipeline (`categorical` columns are one-hot encoded against the fixed `Fair` / `Mostly Cloudy` / `Cloudy` / `Other` sky vocabulary). Two loss-weighting knobs live in `train_config.yaml`'s `hourly:` section, used ONLY by this script (not the TFT pipeline):
- `spread_weight` — squared error on rows far from the training mean, in EITHER direction (peaks and troughs), scaled up to `(1+spread_weight)x`.
- `strength_weight` — one-sided: weight ~0 for calm readings, ramping to 1.0 at the strongest reading, as `(actual/max)^strength_weight`. Takes priority over `spread_weight` if both are nonzero.

`simple_feature_sweep.py` does greedy feature selection for this model: screens every station `DegC`/`KPa`/`Hum` column against a fixed base (read live from `train_config.yaml`'s `real_known`+`real_unknown`), then greedily adds up to `MAX_ADDITIONS` more, scored on a held-out split of the training CSV.

**Findings from that sweep, worth knowing before trusting a feature:**
- Humidity (`{station}Hum`) columns are consistently the strongest predictors found, but exist ONLY in the historical training CSVs — `ec_scrape.py`'s live scraper and the SQLite DB schema never capture humidity, so `--source db` can't use it. `--source csv` can, but that's exploration only; not deployable as-is without extending the live scraper + DB schema.
- `{station}KPa` (pressure) columns are on inconsistent scales between the training CSV and the live DB: the bulk historical download reports genuine *station* pressure, while `ec_scrape.py`'s live scrape of `weather.gc.ca/past_conditions` reports *sea-level-adjusted* pressure. For Pemberton (~204m elevation) that's a systematic ~2.5 kPa offset — large enough to push a live inference input several standard deviations outside the training distribution and produce a visibly broken prediction. Treat `KPa` features as unsafe for live-served models until the two pipelines are reconciled to the same convention.

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
