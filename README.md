# Squamish Wind Model

A wind forecasting system for Squamish, BC. Scrapes weather data from Environment Canada and Squamish Windsports, trains a Temporal Fusion Transformer (TFT) model, and generates an hourly (8-hour) `squamishSpeed` wind forecast.

---

## Workflow Overview

```
collect_data.py  →  build_dataset.py  →  tft_model.py  →  forecast.py
  (historical)       (training CSV)      (TFT model)      (predictions)
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

## 2. Build the Training Dataset

Builds `training_data/hourly_database.csv` from the downloaded raw data.

```bash
python build_dataset.py
```

No flags.

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

All model hyperparameters and feature lists live in `train_config.yaml` — edit this file, not the Python source. The `data.mask_intervals` section excludes known-bad sensor periods at training time.

Key config sections (all under `hourly:`):
- Feature lists (`real_known`, `real_unknown`, `categorical`)
- Sequence lengths (`encoder_length`, `prediction_length`)
- `val_full_data` / `val_predict_mode` — validation dataset construction
- `spread_weight` / `strength_weight` — `simple_model.py`-only loss weighting (not read by the TFT)

---

## Dependencies

Requires Python with `pytorch-forecasting`, `lightning`, `torch`, `pandas`, `numpy`, `selenium` + Chrome (for SWS scraping), `requests`, `beautifulsoup4`, `thefuzz`, `python-dotenv`, `pytz`.

See `environment_pi.yml` for the full pinned environment.
