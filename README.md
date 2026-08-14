# Squamish Wind Model

A wind forecasting system for Squamish, BC. Scrapes weather data from Environment Canada and Squamish Windsports, trains Temporal Fusion Transformer (TFT) models, and generates an hourly (8-hour) `squamishSpeed` wind forecast plus daily (5-day) models for the 2pm snapshot of `squamishSpeed` and two derived 0-5 steadiness scores.

---

## Workflow Overview

```
collect_data.py  →  build_dataset.py  →  tft_model.py        →  forecast.py
  (historical)     (hourly training CSV)  (hourly TFT model)     (predictions)
                          ↓                        ↑
                   build_dataset.py         update_data.py
                    --mode daily            (live DB update)
                          ↓
              (daily training CSV) → tft_daily_model.py
                                       (daily TFT models)
```

---

## 1. Collect Historical Data

Downloads historical Environment Canada station CSVs and Squamish Windsports wind data, and initialises the SQLite database.

```bash
python collect_data.py                        # full run: EC CSVs + SWS data + DB init
python collect_data.py --ec-only              # only download EC station CSVs
python collect_data.py --sws-only             # only download SWS wind data (requires Selenium + Chrome)
python collect_data.py --init-db              # only initialise the SQLite DB
```

| Flag | Description |
|------|-------------|
| `--ec-only` | Download EC station CSVs only |
| `--sws-only` | Download SWS wind data only |
| `--init-db` | Initialise SQLite DB only |
| `--ec-end MonYYYY` | Last month to download for EC (e.g. `Jun2026`) |
| `--sws-start YYYY-MM-DD` | SWS data start date |
| `--sws-end YYYY-MM-DD` | SWS data end date (default: today) |

---

## 2. Build the Training Datasets

Builds `training_data/hourly_database.csv` from the downloaded raw data, and `training_data/daily_database.csv` (2pm snapshot + two derived steadiness scores) from that hourly CSV.

```bash
python build_dataset.py                 # builds both
python build_dataset.py --mode hourly   # only hourly_database.csv
python build_dataset.py --mode daily    # only daily_database.csv — requires hourly_database.csv to already exist
```

| Flag | Description |
|------|-------------|
| `--mode hourly\|daily\|both` | Which dataset to build (default: `both`) |

---

## 3. Train the Model

Trains a TFT model for `squamishSpeed` from the built CSV, and can also predict + plot against the live DB or the CSV.

```bash
python tft_model.py --train                        # train + save models/tft{target}HourlyCheckpoint.ckpt
python tft_model.py --test [--source db|csv]        # predict + plot
python tft_model.py                                 # train then test
```

| Flag | Description |
|------|-------------|
| `--train` | Train and save the model |
| `--test` | Predict against DB or CSV and plot |
| `--epochs N` | Override `max_epochs` from `train_config.yaml` |
| `--start YYYY-MM-DD` | Test start date (default: all available data) |
| `--end YYYY-MM-DD` | Test end date, inclusive (default: no upper bound) — use with `--start` to bound a small window for fast iteration |
| `--source db\|csv` | Where `--test` pulls data from: live SQLite DB (default), or the full training CSV — use `csv` for features not in the live DB (e.g. humidity) |
| `--save` | Save plot to `forecasts/` instead of showing it |

Checkpoint naming: `models/tft{target}HourlyCheckpoint.ckpt`, dataset metadata: `models/{target}_training_dataset_hourly.pkl`

### Plain NN baseline

A simpler same-timestep regression model (no encoder/decoder, no quantile loss) for sanity-checking feature choices independent of the TFT. Reads the same `real_known`/`real_unknown`/`categorical` lists from `train_config.yaml`.

```bash
python simple_model.py --train                   # train + save models/simple_nn_hourly.pt
python simple_model.py --test [--source db|csv]   # predict + plot
python simple_feature_sweep.py                    # greedy feature selection against the current base features
```

---

## 3b. Train a Daily Model

Trains a TFT model for one of three daily targets from `training_data/daily_database.csv` — the raw 2pm `squamishSpeed`, or one of two derived 0-5 steadiness scores (`speed_steadiness`, `direction_steadiness`; see Configuration below). All three share the same `daily:` config section (features, hyperparameters) — only the label column differs.

```bash
python tft_daily_model.py --train [--target NAME]                      # train + save models/tft{target}DailyCheckpoint.ckpt
python tft_daily_model.py --test [--target NAME] [--source db|csv]     # predict + plot
python tft_daily_model.py [--target NAME]                              # train then test
```

| Flag | Description |
|------|-------------|
| `--train` | Train and save the model |
| `--test` | Predict against DB or CSV and plot |
| `--epochs N` | Override `max_epochs` from `train_config.yaml` |
| `--target NAME` | Which `daily: targets` entry to train/test (default: first entry, `squamishSpeed`). `--test --target all` plots every target stacked in one figure |
| `--start` / `--end YYYY-MM-DD` | Bound `--test` to a date window |
| `--source db\|csv` | Where `--test` pulls data from. The steadiness targets only exist in the CSV, not the live DB — required to be `csv` for those |
| `--save` | Save plot to `forecasts/` instead of showing it |

**Caveat:** `--test` currently feeds observed (not forecast) station temperatures into the known-future window, so it's a "perfect foresight" backtest, not a true live-inference test — `forecast.py` has no daily path yet.

### Plain NN baseline / feature sweeps (daily)

```bash
python simple_daily_model.py --train                     # daily squamishSpeed only, no --target flag
python simple_daily_model.py --test [--source csv|db]
python simple_feature_sweep.py --mode daily [--max-additions N]   # greedy feature selection for simple_daily_model.py
python tft_daily_sweep.py           # feature sweep using real TFT training runs instead of the MLP proxy above
python tft_daily_weight_sweep.py    # tests calm-day training-loss deweighting (found not to help; diagnostic only)
```

---

## 4. Update Live Data

Pulls the latest EC and SWS observations into the SQLite DB without generating a forecast. Run this to keep the DB current between forecasts.

```bash
python update_data.py
```

No flags — runs a full DB update.

---

## 5. Run Forecasts

Updates the DB with latest data, then generates CSV and JSON forecast output.

```bash
python forecast.py
```

No flags. Output files: `forecasts/hourly_speed_predictions.{csv,json}`

---

## Configuration

All model hyperparameters and feature lists live in `train_config.yaml` — edit this file, not the Python source. The top-level `data.mask_intervals` section excludes known-bad sensor periods at training time (applies to both hourly and daily).

Key config sections, each present under both `hourly:` and `daily:`:
- Feature lists (`real_known`, `real_unknown`, `categorical`)
- Sequence lengths (`encoder_length`, `prediction_length`)
- `val_full_data` / `val_predict_mode` — validation dataset construction
- `spread_weight` / `strength_weight` — `simple_model.py`/`simple_daily_model.py`-only loss weighting (not read by either TFT)

`daily: targets` lists the three daily models: `squamishSpeed` (raw 2pm speed), and two 0-5 scores computed by `build_dataset.py` from each day's sailable hours (`squamishSpeed > 15kt`) — `speed_steadiness` (gust/lull spread relative to speed) and `direction_steadiness` (circular variance of direction, needs ≥2 sailable hours). Both steadiness scores live only in `training_data/daily_database.csv`, never in the live DB.

---

## Dependencies

Requires Python with `pytorch-forecasting`, `lightning`, `torch`, `pandas`, `numpy`, `selenium` + Chrome (for SWS scraping), `requests`, `beautifulsoup4`, `thefuzz`, `python-dotenv`, `pytz`.

See `environment_pi.yml` for the full pinned environment.
