# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A wind forecasting system for Squamish, BC. It scrapes weather data from Environment Canada and Squamish Windsports, stores it in SQLite, and trains Temporal Fusion Transformers (TFT) for two horizons: an hourly (8-hour) `squamishSpeed` forecast, and a daily (5-day) forecast of the 2pm snapshot — raw `squamishSpeed` plus two derived 0-5 steadiness scores (`speed_steadiness`, `direction_steadiness`).

## Running the System

### Core workflow

```bash
# 1. Download historical EC station CSVs + SWS wind data + initialise DB (first-time / refresh)
python collect_data.py
python collect_data.py --ec-only        # only EC station CSVs (full rebuild → web_data/; reports sparse months, doesn't re-fetch)
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
#           --horizons 1,4 (comma-separated hours-ahead to plot; each line is a fixed
#           horizon from a rolling stride=1 window, so this is slower than a single pass —
#           use --start/--end to keep the run fast)

# 3b. Train a daily model — one model per target (squamishSpeed / speed_steadiness / direction_steadiness)
python tft_daily_model.py --train [--target NAME]                   # default target: squamishSpeed (train_config.yaml daily: targets[0])
python tft_daily_model.py --test [--target NAME] [--source db|csv]  # predict + plot; --target all plots every target stacked in one figure
# steadiness targets only exist in the CSV (not the live DB) — use --source csv for those

# 4. Update the SQLite DB with the latest EC + SWS observations (no forecast generated)
python update_data.py

# 5. Run the forecast (updates DB then generates CSV + JSON output)
python forecast.py                      # both hourly and daily
python forecast.py --mode hourly        # forecasts/hourly_speed_predictions.{csv,json} only
python forecast.py --mode daily         # forecasts/daily_speed_predictions.{csv,json} only (all 3 daily targets — see Daily below)
python forecast.py --skip-features comoxHum,pamHum,vancouverHum
    # Temporary bridging override: named columns get a loudly-logged 0 placeholder for any
    # encoder-window gap instead of failing (e.g. a feature whose live capture just started,
    # like Hum right after the DB migration — see Hourly's Key modules note below). Scoped
    # ONLY to the columns you name; a gap in anything else still fails loudly as normal.
python forecast.py --no-update          # skip update_db() (the EC + SWS scrape), predict from the DB as-is —
                                         # for a cron setup where a separate job already refreshes the DB
```

`collect_data.py`'s EC bulk-download path validates rather than trusting each monthly CSV blindly, mirroring the same philosophy as the live daily-forecast scrape (see Daily below): `_fetch_station_month()` checks the response looks EC-shaped before parsing AND that the parsed columns are actually present (retrying up to 3x, then giving up and returning empty — callers just skip that month); `_blank_implausible_temps()` blanks any `Temp (°C)` value outside -40..50°C (catches a value that's numeric but wrong, which `pd.to_numeric(errors='coerce')` alone can't); `_sparse_months()` flags any calendar month with <50% non-`NaN` temperature coverage. `--ec-update` treats a sparse *recent* month as retry-worthy (EC likely hasn't finished publishing it yet) and re-fetches it within the same run; `--ec-only`'s full rebuild only reports sparse months found across the whole history, since retrying already-settled old data is unlikely to change anything — a genuine gap in EC's own records needs a human to investigate, not a retry loop.

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
web_data/{station}.csv + web_data/sws_wind_database.csv + weather_data_hourly.db
    ↓ (build_dataset.py --mode hourly)
training_data/hourly_database.csv
    ↓ (tft_model.py --train)                    ↓ (build_dataset.py --mode daily — derives ONLY from hourly_database.csv, no raw web_data/ reads)
models/tft*HourlyCheckpoint.ckpt             training_data/daily_database.csv
    ↓ (forecast.py --mode hourly)                ↓ (tft_daily_model.py --train --target NAME)
forecasts/hourly_speed_predictions.{csv,json}  models/tft{target}DailyCheckpoint.ckpt
                                                  ↓ (forecast.py --mode daily — all 3 daily targets)
                                                forecasts/daily_speed_predictions.{csv,json}
```

Training reads the CSV snapshot. Hourly inference reads live data from the SQLite DB via `update_data.py`. Daily inference also reads the DB — including deriving the steadiness targets live from full hourly history, not just CSV — plus a fresh (never persisted) scrape of EC's multi-day temp forecast on every run (see Daily below).

## Architecture

### Hourly (production forecast)

**One TFT model, hourly, single target** (`squamishSpeed` — see `targets` in `train_config.yaml`'s `hourly:` section). Encoder/prediction window lengths are set by `encoder_length`/`prediction_length` in that same section (under active hyperparameter tuning — check the YAML for current values). Trained with `RMSE` loss for a single point forecast (`output_size=1`) — no quantile outputs.

**Column naming convention:**
- `squamishSpeed` / `squamishGust` / `squamishLull` / `squamishDirection` / `squamishDegC` — SWS sensor at the Squamish spit
- `{station}DegC` / `{station}KPa` / `{station}Hum` / `{station}Sky` — EC weather stations

The SQLite DB stores SWS columns under legacy names (`speed`, `gust`, `lull`, `direction`, `temperature`). `update_data.py` applies `_DB_COL_RENAME` when reading from the DB so all downstream code uses the `squamish*` convention.

**Feature selection finding (from `simple_feature_sweep.py`):** station temperature (`DegC`) and humidity (`Hum`) columns are consistently the strongest predictors; pressure (`KPa`) columns add little to nothing within the training CSV, and are additionally unsafe to serve live (see the sea-level-vs-station-pressure mismatch noted under Plain NN Baseline below). Humidity (`{station}Hum`) is now scraped live for all 8 stations by `ec_scrape.py`'s `pull_past_hrs_weather()` and stored in the `weather` table — added specifically because `train_config.yaml`'s hourly `real_unknown` already includes `comoxHum`/`pamHum`, and without live data those columns silently made `--source db` inference impossible (a `KeyError`, not a graceful fallback).

**Known real features** (available in the forecast window): none active by default — see `real_known` in `train_config.yaml`.
**Unknown real features** (encoder past only): station DegC/KPa/Hum columns — see `real_unknown` in `train_config.yaml` for current active set.

There is no derived scoring layer (no steadiness/gust-relative/quality-score computation anywhere in the pipeline) — training and forecast output are both raw `squamishSpeed` in knots.

**Key modules:**
- `config.py` — `HourlyConfig` dataclass; populated from `train_config.yaml`'s `hourly:` section via `from_yaml()`
- `tft_model.py` — the whole TFT pipeline in one file: `tft_with_ignore` (checkpoint-safe model subclass), `_build_dataset()`, `_apply_mask_intervals()`, `train()`, `test()` (predicts against `--source db|csv` and plots). Also the shared source of `_apply_mask_intervals()` for `simple_model.py`/`simple_feature_sweep.py`.
- `update_data.py` — `update_db()` pulls latest EC + SWS data into SQLite (and calls `collect_data.migrate_schema()` first — see below); `get_conditions_table_hourly()` calls it then builds the inference DataFrame; runnable standalone for a DB-only update. Merges only DB history (no forecast scrape). `_DB_COL_RENAME` (legacy `speed`/`gust`/`lull`/`direction`/`temperature` → `squamish*`) is applied in `update_db()`'s write path and in `get_conditions_table_daily()`'s read path — `get_conditions_table_hourly()` doesn't need it since the DB's own column names are already `squamish*` post-`migrate_schema()`. Anchors its window to `min(wall-clock now, latest DB row)` rather than blind wall-clock time — hourly's `encoder_length` is short (see `train_config.yaml`'s `hourly:` section for the current value), and EC's live feed routinely lags wall-clock by more than that, so anchoring to raw "now" could push the entire encoder window past the last real reading (zero real data → downstream crash) whenever that lag exceeded the encoder length. Anchoring to whichever is earlier means a live-data lag degrades to "predict from the latest data we actually have" instead of crashing. Both `get_conditions_table_hourly`/`get_conditions_table_daily` query `WHERE datetime >= start` (not `>`) — `pd.date_range(start=start, ...)` is inclusive of `start`, and since `start` is now anchored to an actual observed timestamp (not arbitrary wall-clock time), it lands exactly on a real row routinely; a strict `>` silently dropped that row on every such call.
- `collect_data.py` — `migrate_schema(conn)` brings an existing `weather` table up to the current schema in place: `_migrate_add_hum_columns()` (`ALTER TABLE ADD COLUMN`, additive/lossless) and `_migrate_rename_legacy_sws_columns()` (`ALTER TABLE RENAME COLUMN` from the pre-migration `speed`/`gust`/`lull`/`direction`/`temperature` names, lossless — see git history "Rename SWS columns in SQLite DB to squamish* convention"). Runs on every `update_db()` call, not just `--init-db` — a `weather` table copied in from a different deployment (e.g. the Raspberry Pi, running older code) can be on an older schema at any time, and self-heals the moment anything touches the DB instead of crashing with a confusing `KeyError`.
- `forecast.py` — `_warn_missing(data, features, label, prediction_length)` reports NaN counts + max consecutive gap, but **only within the encoder window** — it deliberately excludes the prediction-horizon rows from what it reports, since those are structurally always `NaN` (see below) and not a data-quality issue worth flagging. "ALL N encoder-window values missing" indicates a scraper or DB failure (or, for a freshly-added feature like `Hum`, simply not enough history captured yet). No interpolation/ffill/bfill of any kind — the only fill left is a placeholder (`0`) applied *exclusively* to each column's prediction-horizon rows (`targets` and `real_unknown`, never `real_known`), since those are structurally always `NaN` regardless of data quality (the target is what's being forecast; `real_unknown` is by definition not known in the future) and are never read by the model there — pytorch_forecasting just requires a non-`NaN` tensor. A `NaN` anywhere in the *encoder* window, or in `real_known`'s future window, is a genuine data gap, gets reported by `_warn_missing()`, and raises loudly at `TimeSeriesDataSet` construction rather than being papered over. `get_conditions_table_hourly`/`get_conditions_table_daily` in `update_data.py` still keep one light fill of their own: a single-step (`limit=1`) linear interpolation immediately after the DB read, bridging a lone missing hour — this is the one exception, kept deliberately. Outputs raw `squamishSpeed` predictions only, under the website's expected legacy field name: `_OUTPUT_COL_RENAME` (built as the inverse of `update_data.py`'s `_DB_COL_RENAME`) renames `squamishSpeed` → `speed` in `hourly_speed_predictions.{csv,json}` — the rename happens only on the already-finished output DataFrame right before it's written; every internal column (`_hcfg.targets`, model I/O, DB reads) still uses `squamishSpeed` throughout. `_DAILY_OUTPUT_COL_RENAME` extends this for `daily_speed_predictions.{csv,json}`, additionally renaming `speed_steadiness`/`direction_steadiness` → the pipeline's pre-rebuild names `speed_score`/`direction_score` (same underlying scores, renamed across a rebuild) — `hours_above_20`, an old field with no current equivalent, is simply absent since it was never a `daily: targets` column to begin with.
- `build_dataset.py` — builds `training_data/hourly_database.csv` from the raw station/SWS CSVs; also exports `wind_to_hourly_index()` (±10min SWS-to-hour merge), reused by `update_data.py` for the live DB path.
- `ec_scrape.py` — all live EC scraping: `pull_past_hrs_weather()`; also exports `normalize_sky_series()`. (`pull_forecast_hourly()` and `pull_forecast_daily()` exist but are no longer called by anything.)
- `sws_pull.py` — Selenium-based SWS wind data fetch; `get_sws_df(dates)` for arbitrary date lists; `update_sws_csv()` for incremental append since last CSV entry

### Daily (multi-day speed + steadiness)

**Three TFT models, daily, one target each**, all sharing one blueprint — the `daily:` section's `real_known`/`real_unknown`/hyperparameters in `train_config.yaml` (5-day encoder, 5-day prediction, quantile outputs). `tft_daily_model.py --target NAME` selects which target's checkpoint to train/test; only the label column differs between the three models. Model capacity is deliberately smaller than hourly's (`hidden_size: 16` vs `64`) since daily has ~14x less data.

**Targets** (`train_config.yaml`'s `daily: targets`):
- `squamishSpeed` — the raw 2pm wind speed snapshot (default target, index 0)
- `speed_steadiness` — 0-5 score (5=steadiest) from the gust/lull spread relative to speed, `(squamishGust - squamishLull) / squamishSpeed`, averaged over each day's sailable hours (`squamishSpeed > 15kt`) and mapped through fixed p10/p90 calibration constants. Days with no sailable hours score 0.
- `direction_steadiness` — 0-5 score (5=steadiest) from the circular variance of `squamishDirection` over each day's sailable hours (needs ≥2 such hours to have measurable spread; fewer scores 0). Circular variance (not plain std) because direction wraps at 360°.

Both steadiness scores are computed by `build_dataset.py`'s `add_speed_steadiness()`/`add_direction_steadiness()`. They're written to `training_data/daily_database.csv` for training, but are ALSO computed live by `update_data.py`'s `get_conditions_table_daily()` — it fetches the full hourly (not just 2pm) `weather` table history for the window and calls the same two functions directly (they now accept an in-memory `hourly=` DataFrame as an alternative to reading `hourly_path`'s CSV), so `forecast.py --mode daily` forecasts all three targets, not just `squamishSpeed`. `tft_daily_model.py --test`, unlike `forecast.py`, still requires `--source csv` for these two targets (raises a clear error if `--source db` is passed) — that backtest path hasn't been wired to the live computation.

**Feature selection finding (from `tft_daily_sweep.py`):** the temperature-station base beat every one of 21 candidate additions tried (other DegC stations, all KPa, all Hum, `squamishDegC`) — echoing the same "adding more hurts" finding as hourly. `real_known` is that base (5 station `DegC` columns, since EC's 6-day forecast covers them); `real_unknown` currently holds `KPa`/`Hum`/`squamishDegC`/`pamDegC` (see `train_config.yaml`'s `daily:` section for the exact, current lists — they shift as retraining happens, e.g. `pamDegC` just moved here from `real_known`).

`tft_daily_weight_sweep.py` tested deweighting calm days (`<12kt`) during training, to see if it sharpens peak-calling — it monotonically hurt val_loss at every value tried (1.0 baseline → 0.0 full exclusion), so it was NOT adopted; `tft_daily_model.py`'s `train()` has no deweighting option. `tft_model.py`'s shared `_build_dataset()` still has inert opt-in support for a `weight` column (exercised only by that sweep script) if this is ever revisited.

**CAVEAT:** `tft_daily_model.py --test` still feeds the `real_known` window from OBSERVED station temperatures (DB or CSV) — it's a "perfect foresight" backtest, not a true live-inference test, and that hasn't changed. `forecast.py --mode daily`, however, is a genuine live-inference path: `update_data.py`'s `_pull_daily_forecast()` scrapes EC's actual multi-day temp/sky forecast (`ec_scrape.pull_forecast_daily()`) fresh on every call — nothing is persisted to SQLite, since a forecast is only useful to the run that requested it — and `get_conditions_table_daily()` splices it into `real_known`'s future window in-memory (observed DB data still wins for historical/encoder rows via `combine_first`). The scrape is validated, not trusted blindly: a row-count mismatch (usually one station returning misaligned dates, since `pull_forecast_daily()`'s per-station inner join then misaligns everyone) discards the whole pull; a missing station is left absent; an individual missing/out-of-range temperature (outside -40..50°C) is blanked. **None of that is backfilled** — `forecast.py` no longer fills any `real_known` gap (see Hourly's Key modules note above), so a blank/missing/discarded value there now means the run fails at `TimeSeriesDataSet` construction. This is a *permanent, every-run* failure for `pamDegC` specifically — it's also `real_known`, but EC has no daily-forecast page for that station at all, so its future window is always 100% `NaN`. Fixing this needs `pamDegC` moved to `real_known`'s `real_unknown` counterpart in `train_config.yaml` and the daily models retrained (in progress). `forecast.py --mode daily` forecasts every target in `daily: targets` that `get_conditions_table_daily()` produces a column for (currently all three — see the steadiness note above); any target it doesn't produce is skipped with a printed warning rather than erroring.

**Key modules:**
- `config.py` — `DailyConfig` dataclass, same shape as `HourlyConfig`; `training_cutoff()` differs (daily's is far thinner, hence `val_full_data: true` by default)
- `tft_daily_model.py` — mirrors `tft_model.py`'s shape, reusing its `tft_with_ignore`/`_build_dataset`/`_apply_mask_intervals`; `--target` selects the target (defaults to `daily: targets[0]`), `--test --target all` plots every target stacked in one figure
- `simple_daily_model.py` — plain-NN baseline for daily `squamishSpeed` only (no `--target` flag, unlike `tft_daily_model.py`); mirrors `simple_model.py`
- `build_dataset.py` — `build_daily()` derives entirely from `training_data/hourly_database.csv` (keeps each day's 2pm row), then calls `add_speed_steadiness()`/`add_direction_steadiness()`
- `tft_daily_sweep.py` / `tft_daily_weight_sweep.py` — standalone diagnostic scripts (feature sweep and calm-day-deweighting sweep, respectively); not imported by anything else
- `update_data.py` — `_pull_daily_forecast()` scrapes and validates EC's daily forecast (never persisted); `get_conditions_table_daily()` merges it with `weather` table history into the inference DataFrame
- `forecast.py` — `run_daily()` / `_prepare_daily()` / `_predict_daily_target()` generate `forecasts/daily_speed_predictions.{csv,json}` for every `daily: targets` column present in the data (currently all three), mirroring the hourly functions

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
- Humidity (`{station}Hum`) columns are consistently the strongest predictors found. They used to exist ONLY in the historical training CSVs, but `ec_scrape.py`'s `pull_past_hrs_weather()` now scrapes `{station}Hum` live for all 8 stations and `collect_data.py`'s `weather` table schema has matching columns (added via `_migrate_add_hum_columns()`, an `ALTER TABLE ADD COLUMN` migration safe to run against an existing DB — it only adds NULL-filled columns, never touches prior rows). `--source db` can use it going forward; historical DB rows predating the migration simply have `NULL` humidity. `collect_data.py`'s `migrate_schema(conn)` bundles this with `_migrate_rename_legacy_sws_columns()` (renames a pre-migration `weather` table's raw `speed`/`gust`/`lull`/`direction`/`temperature` columns to the current `squamish*` convention — see git history "Rename SWS columns in SQLite DB to squamish* convention"), and `update_data.py`'s `update_db()` calls it on every run, not just via `collect_data.py --init-db` — a `weather` table swapped in from a different deployment (e.g. the Raspberry Pi, running older code) can be on an older schema at any time, and self-heals the next time anything touches the DB rather than crashing with a confusing `KeyError`.
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
