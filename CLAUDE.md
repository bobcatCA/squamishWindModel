# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A wind forecasting system for Squamish, BC. It scrapes weather data from Environment Canada and Squamish Windsports, stores it in SQLite, and trains Temporal Fusion Transformers (TFT) for two horizons: an hourly (8-hour) `squamishSpeed` forecast, and a daily (5-day) forecast of the 2pm snapshot — raw `squamishSpeed` plus two derived 0-5 steadiness scores (`speed_steadiness`, `direction_steadiness`).

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

# 2. Build the training datasets from downloaded raw data
python build_dataset.py                 # builds both hourly_database.csv and daily_database.csv
python build_dataset.py --mode hourly   # only training_data/hourly_database.csv
python build_dataset.py --mode daily    # only training_data/daily_database.csv (derives from hourly_database.csv — run hourly first)

# 3. Train the hourly model
python tft_model.py --train                        # targets defined in train_config.yaml (currently squamishSpeed only)
python tft_model.py --test [--source db|csv]        # predict + plot
# Optional: --epochs N   --start/--end YYYY-MM-DD (bound --test to a window)  --save

# 3b. Train a daily model — one model per target (squamishSpeed / speed_steadiness / direction_steadiness)
python tft_daily_model.py --train [--target NAME]                   # default target: squamishSpeed (train_config.yaml daily: targets[0])
python tft_daily_model.py --test [--target NAME] [--source db|csv]  # predict + plot; --target all plots every target stacked in one figure
# steadiness targets only exist in the CSV (not the live DB) — use --source csv for those

# 4. Update the SQLite DB with the latest EC + SWS observations (no forecast generated)
python update_data.py

# 5. Run the forecast (updates DB then generates CSV + JSON output) — hourly only, no daily path yet
python forecast.py
```

### Utility scripts

```bash
# Plain feedforward-NN baseline (no encoder/decoder, no quantile loss) — same
# real_known/real_unknown/categorical config as the TFT, for isolating whether
# the configured features explain the target at all
python simple_model.py --train                                   # hourly squamishSpeed
python simple_model.py --test [--source db|csv] [--start DATE] [--save]
python simple_daily_model.py --train                              # daily squamishSpeed (fixed target, no --target flag)
python simple_daily_model.py --test [--source csv|db] [--start DATE] [--save]

# Greedy feature selection, against the currently active real_known/real_unknown as the base
python simple_feature_sweep.py                          # hourly (default)
python simple_feature_sweep.py --mode daily [--max-additions N]

# TFT feature sweep for the daily models (uses real training runs, not the MLP proxy above) —
# base = current daily: real_known/real_unknown, screens every other station column as a single addition
python tft_daily_sweep.py

# Tests whether deweighting calm days (<12kt) during daily TFT training helps — it doesn't
# (monotonically hurts val_loss at every value tried); kept as a diagnostic, not wired into
# tft_daily_model.py's train()
python tft_daily_weight_sweep.py
```

## Configuration

All model hyperparameters and feature lists live in `train_config.yaml` — it is the single source of truth, under separate `hourly:` and `daily:` sections (both loaded via `config.py`'s `HourlyConfig`/`DailyConfig`, sharing a `_BaseConfig` parent). The Python dataclass defaults in `config.py` are fallbacks only; always edit the YAML. The top-level `data.mask_intervals` section excludes known-bad sensor periods at training time without altering the CSVs, and applies to both hourly and daily.

Notable YAML keys beyond features and sequence lengths: `val_full_data` / `val_predict_mode` control validation dataset construction; `spread_weight` / `strength_weight` are `simple_model.py`/`simple_daily_model.py`-only loss-weighting knobs, not read by either TFT pipeline.

Checkpoint naming: `models/tft{target}HourlyCheckpoint.ckpt` / `models/tft{target}DailyCheckpoint.ckpt` (e.g., `models/tftsquamishSpeedHourlyCheckpoint.ckpt`, `models/tftspeed_steadinessDailyCheckpoint.ckpt`)
Dataset metadata naming: `models/{target}_training_dataset_hourly.pkl` / `models/{target}_training_dataset_daily.pkl`

## Data Flow

```
Environment Canada + Squamish Windsports
    ↓ (collect_data.py — historical; ec_scrape.py + sws_pull.py — live)
web_data/{station}.csv + web_data/sws_wind_database.csv + web_data/weather_data_hourly.db
    ↓ (build_dataset.py --mode hourly)
training_data/hourly_database.csv
    ↓ (tft_model.py --train)                    ↓ (build_dataset.py --mode daily — derives ONLY from hourly_database.csv, no raw web_data/ reads)
models/tft*HourlyCheckpoint.ckpt             training_data/daily_database.csv
    ↓ (forecast.py)                              ↓ (tft_daily_model.py --train --target NAME)
forecasts/hourly_speed_predictions.{csv,json}  models/tft{target}DailyCheckpoint.ckpt  (no forecast.py path yet — see Architecture below)
```

Training reads the CSV snapshot. Hourly inference reads live data from the SQLite DB via `update_data.py`; the daily steadiness targets have no DB path (CSV-only, see below).

## Architecture

### Hourly (production forecast)

**One TFT model, hourly, single target** (`squamishSpeed` — see `targets` in `train_config.yaml`'s `hourly:` section). 12-hour encoder, 8-hour prediction, quantile outputs (Q1–Q7).

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

### Daily (multi-day speed + steadiness)

**Three TFT models, daily, one target each**, all sharing one blueprint — the `daily:` section's `real_known`/`real_unknown`/hyperparameters in `train_config.yaml` (5-day encoder, 5-day prediction, quantile outputs). `tft_daily_model.py --target NAME` selects which target's checkpoint to train/test; only the label column differs between the three models. Model capacity is deliberately smaller than hourly's (`hidden_size: 16` vs `64`) since daily has ~14x less data.

**Targets** (`train_config.yaml`'s `daily: targets`):
- `squamishSpeed` — the raw 2pm wind speed snapshot (default target, index 0)
- `speed_steadiness` — 0-5 score (5=steadiest) from the gust/lull spread relative to speed, `(squamishGust - squamishLull) / squamishSpeed`, averaged over each day's sailable hours (`squamishSpeed > 15kt`) and mapped through fixed p10/p90 calibration constants. Days with no sailable hours score 0.
- `direction_steadiness` — 0-5 score (5=steadiest) from the circular variance of `squamishDirection` over each day's sailable hours (needs ≥2 such hours to have measurable spread; fewer scores 0). Circular variance (not plain std) because direction wraps at 360°.

Both steadiness scores are computed by `build_dataset.py`'s `add_speed_steadiness()`/`add_direction_steadiness()` and only ever written to `training_data/daily_database.csv` — never to the live SQLite DB. `tft_daily_model.py --test` requires `--source csv` for these two targets (raises a clear error if `--source db` is passed).

**Feature selection finding (from `tft_daily_sweep.py`):** the 4-temperature base (`lillooetDegC`, `pembertonDegC`, `vancouverDegC`, `whistlerDegC`) beat every one of 21 candidate additions tried (other DegC stations, all KPa, all Hum, `squamishDegC`) — echoing the same "adding more hurts" finding as hourly. `real_unknown` is empty as a result.

`tft_daily_weight_sweep.py` tested deweighting calm days (`<12kt`) during training, to see if it sharpens peak-calling — it monotonically hurt val_loss at every value tried (1.0 baseline → 0.0 full exclusion), so it was NOT adopted; `tft_daily_model.py`'s `train()` has no deweighting option. `tft_model.py`'s shared `_build_dataset()` still has inert opt-in support for a `weight` column (exercised only by that sweep script) if this is ever revisited.

**CAVEAT:** `tft_daily_model.py --test` currently feeds the `real_known` window from OBSERVED station temperatures (DB or CSV), not genuine EC forecast values — `pull_forecast_daily()` in `ec_scrape.py` isn't wired into either test path yet, so this is a "perfect foresight" backtest, not a true live-inference test. `forecast.py` has no daily path at all yet.

**Key modules:**
- `config.py` — `DailyConfig` dataclass, same shape as `HourlyConfig`; `training_cutoff()` differs (daily's is far thinner, hence `val_full_data: true` by default)
- `tft_daily_model.py` — mirrors `tft_model.py`'s shape, reusing its `tft_with_ignore`/`_build_dataset`/`_apply_mask_intervals`; `--target` selects the target (defaults to `daily: targets[0]`), `--test --target all` plots every target stacked in one figure
- `simple_daily_model.py` — plain-NN baseline for daily `squamishSpeed` only (no `--target` flag, unlike `tft_daily_model.py`); mirrors `simple_model.py`
- `build_dataset.py` — `build_daily()` derives entirely from `training_data/hourly_database.csv` (keeps each day's 2pm row), then calls `add_speed_steadiness()`/`add_direction_steadiness()`
- `tft_daily_sweep.py` / `tft_daily_weight_sweep.py` — standalone diagnostic scripts (feature sweep and calm-day-deweighting sweep, respectively); not imported by anything else

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

`simple_feature_sweep.py` does greedy feature selection for this model: screens every station `DegC`/`KPa`/`Hum` column against a fixed base (read live from `train_config.yaml`'s `real_known`+`real_unknown`), then greedily adds up to `--max-additions` more (default 2), scored on a held-out split of the training CSV. `--mode daily` runs the same sweep against `simple_daily_model.py`/`daily:`'s config instead (default `--max-additions` 99, since daily's base often starts empty).

`simple_daily_model.py` mirrors this same baseline for daily `squamishSpeed`, reading `daily:`'s `real_known`/`real_unknown`/`categorical` — unlike `tft_daily_model.py` it has no `--target` flag (`squamishSpeed` only).

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
