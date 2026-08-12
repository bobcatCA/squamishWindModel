# Squamish Wind Model

A wind forecasting system for Squamish, BC. Scrapes weather data from Environment Canada and Squamish Windsports, trains Temporal Fusion Transformer (TFT) models, and generates hourly (8-hour) and daily (5-day) wind forecasts with quality scores for windsports suitability.

---

## Workflow Overview

```
collect_data.py  →  build_dataset.py  →  train.py  →  forecast.py
  (historical)       (training CSVs)    (TFT models)   (predictions)
                                                  ↑
                                           update_data.py
                                          (live DB update)
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

## 2. Build Training Datasets

Builds `hourly_database.csv` and `daily_database.csv` from the downloaded raw data.

```bash
python build_dataset.py                       # build both hourly and daily
python build_dataset.py --mode hourly
python build_dataset.py --mode daily
```

| Flag | Description |
|------|-------------|
| `--mode hourly\|daily\|both` | Which dataset to build (default: `both`) |

---

## 3. Train Models

Trains TFT models from the built CSVs. Each mode trains 4 separate models (one per target variable).

```bash
python train.py --mode hourly                 # targets: speed, gust, lull, direction
python train.py --mode daily                  # targets: speed, hours_above_20, speed_score, direction_score
```

| Flag | Description |
|------|-------------|
| `--mode hourly\|daily` | Which pipeline to train (**required**) |
| `--epochs N` | Override `max_epochs` from config |
| `--dropout F` | Override dropout rate |
| `--hidden-size N` | Override hidden layer size |
| `--lr F` | Override learning rate |
| `--targets TARGET [...]` | Train only specific targets (e.g. `--targets speed gust`) |
| `--no-lr-find` | Skip automatic learning rate finder |
| `--checkpoint-prefix PREFIX` | Save to named checkpoints (e.g. `tftExp`) instead of overwriting production checkpoints |

Default checkpoint names: `tft{target}{Hourly\|Daily}Checkpoint.ckpt`

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
python forecast.py                            # run both hourly and daily (one DB update)
python forecast.py --mode hourly
python forecast.py --mode daily
```

| Flag | Description |
|------|-------------|
| `--mode hourly\|daily\|both` | Which forecast to generate (default: `both`) |

Output files: `hourly_speed_predictions.{csv,json}`, `daily_speed_predictions.{csv,json}`

---

## Utilities

### Evaluate model accuracy

```bash
python evaluate.py --start 2025-06-20        # evaluate against historical DB from this date
```

### Feature selection sweep

```bash
python feature_selection.py --mode hourly --epochs 2
python feature_selection.py --mode daily  --epochs 3
```

### Plain NN baseline

A simpler same-timestep regression model (no encoder/decoder, no quantile loss) for sanity-checking feature choices independent of the TFT pipeline. Reads the same `real_known`/`real_unknown`/`categorical` lists from `train_config.yaml`.

```bash
python simple_model.py --train                   # train + save models/simple_nn_hourly.pt
python simple_model.py --test [--source db|csv]   # predict + plot (csv exposes Hum features the live DB doesn't have)
python simple_feature_sweep.py                    # greedy feature selection against the current base features
```

---

## Configuration

All model hyperparameters and feature lists live in `train_config.yaml` — edit this file, not the Python source. The `data.mask_intervals` section excludes known-bad sensor periods at training time.

Key config sections:
- Feature lists (`known_reals`, `unknown_reals`)
- Sequence lengths (`encoder_length`, `prediction_length`)
- `val_full_data` / `val_predict_mode` — validation dataset construction

---

## Dependencies

Requires Python with `pytorch-forecasting`, `lightning`, `torch`, `pandas`, `numpy`, `selenium` + Chrome (for SWS scraping), `requests`, `beautifulsoup4`, `thefuzz`, `python-dotenv`, `pytz`.

See `environment_pi.yml` for the full pinned environment.
