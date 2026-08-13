# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A wind forecasting system for Squamish, BC. It scrapes weather data from Environment Canada and Squamish Windsports, stores it in SQLite, and trains a Temporal Fusion Transformer (TFT) to generate an hourly (8-hour) `squamishSpeed` forecast.

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

# 2. Build the training dataset from downloaded raw data
python build_dataset.py                 # builds training_data/hourly_database.csv

# 3. Train the model
python tft_model.py --train                        # targets defined in train_config.yaml (currently squamishSpeed only)
python tft_model.py --test [--source db|csv]        # predict + plot
# Optional: --epochs N   --start/--end YYYY-MM-DD (bound --test to a window)  --save

# 4. Update the SQLite DB with the latest EC + SWS observations (no forecast generated)
python update_data.py

# 5. Run the forecast (updates DB then generates CSV + JSON output)
python forecast.py
```

### Utility scripts

```bash
# Plain feedforward-NN baseline (no encoder/decoder, no quantile loss) — same
# real_known/real_unknown/categorical config as the TFT, for isolating whether
# the configured features explain squamishSpeed at all
python simple_model.py --train
python simple_model.py --test [--source db|csv] [--start DATE] [--save]

# Greedy feature selection for simple_model.py, against the currently active
# real_known/real_unknown as the base
python simple_feature_sweep.py
```

## Configuration

All model hyperparameters and feature lists live in `train_config.yaml` — it is the single source of truth. The Python dataclass defaults in `config.py` are fallbacks only; always edit the YAML. The `data.mask_intervals` section excludes known-bad sensor periods at training time without altering the CSV.

Notable YAML keys beyond features and sequence lengths: `val_full_data` / `val_predict_mode` control validation dataset construction; `spread_weight` / `strength_weight` are `simple_model.py`-only loss-weighting knobs, not read by the TFT.

Checkpoint naming: `models/tft{target}HourlyCheckpoint.ckpt` (e.g., `models/tftsquamishSpeedHourlyCheckpoint.ckpt`)
Dataset metadata naming: `models/{target}_training_dataset_hourly.pkl`

## Data Flow

```
Environment Canada + Squamish Windsports
    ↓ (collect_data.py — historical; ec_scrape.py + sws_pull.py — live)
web_data/{station}.csv + web_data/sws_wind_database.csv + web_data/weather_data_hourly.db
    ↓ (build_dataset.py)
training_data/hourly_database.csv
    ↓ (tft_model.py --train)
models/tft*HourlyCheckpoint.ckpt + models/*_training_dataset_hourly.pkl
    ↓ (forecast.py)
forecasts/hourly_speed_predictions.{csv,json}
```

Training reads the CSV snapshot. Inference reads live data from the SQLite DB via `update_data.py`.

## Architecture

**One TFT model, hourly, single target** (`squamishSpeed` — see `targets` in `train_config.yaml`). 12-hour encoder, 8-hour prediction, quantile outputs (Q1–Q7).

**Column naming convention:**
- `squamishSpeed` / `squamishGust` / `squamishLull` / `squamishDirection` / `squamishDegC` — SWS sensor at the Squamish spit
- `{station}DegC` / `{station}KPa` / `{station}Hum` / `{station}Sky` — EC weather stations

The SQLite DB stores SWS columns under legacy names (`speed`, `gust`, `lull`, `direction`, `temperature`). `update_data.py` applies `_DB_COL_RENAME` when reading from the DB so all downstream code uses the `squamish*` convention.

**Feature selection finding (from `simple_feature_sweep.py`):** station temperature (`DegC`) and humidity (`Hum`) columns are consistently the strongest predictors; pressure (`KPa`) columns add little to nothing within the training CSV, and are additionally unsafe to serve live (see the sea-level-vs-station-pressure mismatch noted under Plain NN Baseline below). Humidity has no live-data path at all.

**Known real features** (available in the forecast window): none active by default — see `real_known` in `train_config.yaml`.
**Unknown real features** (encoder past only): station DegC/KPa/Hum columns — see `real_unknown` in `train_config.yaml` for current active set.

There is no derived scoring layer (no steadiness/gust-relative/quality-score computation anywhere in the pipeline) — training and forecast output are both raw `squamishSpeed` in knots.

**Key modules:**
- `config.py` — `HourlyConfig` dataclass; populated from `train_config.yaml`'s `hourly:` section via `from_yaml()`
- `tft_model.py` — the whole TFT pipeline in one file: `tft_with_ignore` (checkpoint-safe model subclass), `_build_dataset()`, `_apply_mask_intervals()`, `train()`, `test()` (predicts against `--source db|csv` and plots). Also the shared source of `_apply_mask_intervals()` for `simple_model.py`/`simple_feature_sweep.py`.
- `update_data.py` — `update_db()` pulls latest EC + SWS data into SQLite; `get_conditions_table_hourly()` calls it then builds the inference DataFrame; applies `_DB_COL_RENAME` to map legacy DB column names to `squamish*`; runnable standalone for a DB-only update. Merges only DB history (no forecast scrape).
- `forecast.py` — `_warn_missing()` fires before each fill step and prints NaN counts + max consecutive gap per feature; "ALL values missing" indicates a scraper or DB failure. Outputs raw `squamishSpeed` predictions only.
- `build_dataset.py` — builds `training_data/hourly_database.csv` from the raw station/SWS CSVs; also exports `wind_to_hourly_index()` (±10min SWS-to-hour merge), reused by `update_data.py` for the live DB path.
- `ec_scrape.py` — all live EC scraping: `pull_past_hrs_weather()`; also exports `normalize_sky_series()`. (`pull_forecast_hourly()` and `pull_forecast_daily()` exist but are no longer called by anything.)
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
