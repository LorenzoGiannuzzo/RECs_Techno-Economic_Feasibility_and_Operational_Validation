# Input data

The pipeline reads seven input series. Four of them are public and are distributed in `data/public/`; the other three
are the metered consumption of the community, which is confidential and is **not** distributed. This page describes
every file, its origin, and its exact format, so that the confidential files can be replaced either by the original
data (for the authors and for anyone who obtains them) or by a dataset of a different community with the same format.

```
data/
├── public/                         distributed with the repository
│   ├── pv_production_2025.csv
│   ├── zonal_prices_south_2025.csv
│   ├── outdoor_temperature_hourly.csv
│   └── weather_daily.csv
├── confidential/                   NOT distributed (ignored by git)
│   ├── README.md
│   ├── community_loads.csv         (to be provided)
│   └── residential_sample.csv      (to be provided)
└── synthetic/                      generated on demand (ignored by git)
```

## General conventions

- Files are comma-separated, UTF-8, with a header row and a dot as decimal separator.
- Timestamps are written as `YYYY-MM-DD HH:MM` in local time without daylight-saving shifts, that is, every day has
  exactly 24 rows and every file has exactly **8760 consecutive hourly rows**; the value of a row refers to the hour
  that starts at the timestamp (e.g., `2025-06-01 13:00` is the energy between 13:00 and 14:00).
- Energy values are in kWh per hour, prices in €/MWh, temperatures in °C.
- Empty cells are not accepted. Hours without a measurement are stored as `0` and are interpolated by the pipeline
  (this is how the meters store the hour skipped by the daylight-saving change, 2025-03-30 02:00).
- The loader (`src/rec_pipeline/data.py`) checks the columns, the number of rows, and the timestamps of every file and
  stops with an explicit message if anything is not as described here.

## Two calendars

| Calendar | Period | Used by |
|---|---|---|
| Simulated year | 2025-01-01 00:00 – 2025-12-31 23:00 | PV production, zonal prices, all the dispatch simulations |
| Measurement campaign | 2024-10-01 00:00 – 2025-09-30 23:00 | metered loads, residential sample, outdoor temperature |

The metered year is mapped onto the simulated year by the pipeline (Section 3.1 of the paper): January–September 2025
are used as measured, October–December 2024 are moved to October–December 2025 preserving the day of the week
(2024-10-02 and 2025-10-01 are both Wednesdays), and the last day of 2025 reuses 2024-12-31. The forecasting models
work on the chronological measurement calendar, and their out-of-sample period (January–September 2025) coincides
with the first 6552 hours of the simulated year.

## Public data (`data/public/`)

### `pv_production_2025.csv`

| Column | Unit | Description |
|---|---|---|
| `timestamp` | – | hourly, simulated year 2025 |
| `pv_kwh` | kWh | AC energy delivered by the PV plant in the hour |

Hourly production of the ground-mounted plant of the case study (1037 kWp DC, 1000 kW AC), simulated with PVsyst as
described in Section 3.2 of the paper (annual yield 1666.0 MWh, 1607 kWh/kWp). This series is the corrected one: the
spreadsheet used in the thesis stored the MILP input with a one-hour delay, caused by the parsing of the header of the
PVsyst export, and the delay is removed here.

### `zonal_prices_south_2025.csv`

| Column | Unit | Description |
|---|---|---|
| `timestamp` | – | hourly, simulated year 2025 |
| `price_eur_mwh` | €/MWh | day-ahead zonal price of the South (SUD) zone |

Hourly zonal prices of the Italian day-ahead market (MGP) for 2025, published by the market operator GME
(Gestore dei Mercati Energetici, https://www.mercatoelettrico.org). They are used for the RID revenue, for the
variable component of the incentive tariff, and for the arbitrage strategy.

### `outdoor_temperature_hourly.csv`

| Column | Unit | Description |
|---|---|---|
| `timestamp` | – | hourly, measurement campaign (2024-10-01 – 2025-09-30) |
| `temperature_c` | °C | outdoor air temperature at the site |

Input of the MLP load forecaster. The measured temperature is used, which corresponds to assuming a perfect weather
forecast (Section 5.5 of the paper).

### `weather_daily.csv`

| Column | Unit | Description |
|---|---|---|
| `date` | – | `YYYY-MM-DD`, the 365 days of the measurement campaign |
| `t_avg_c`, `t_max_c`, `t_min_c` | °C | mean, maximum, and minimum daily outdoor temperature |

Inputs of the random forest day-type classifier, together with the day of the week, which the pipeline derives from
the date.

## Confidential data (`data/confidential/`, not distributed)

The consumption of the community was provided by the local distribution system operator for the members of the
prospective REC and is subject to confidentiality, as stated in the Data Availability section of the paper. The two
files below are therefore excluded from the repository by `.gitignore`. To reproduce the published results, place the
original files in `data/confidential/` (or in any folder indicated by the environment variable
`REC_CONFIDENTIAL_DIR`). To apply the pipeline to another community, prepare two files with the same format.

### `community_loads.csv`

| Column | Unit | Description |
|---|---|---|
| `timestamp` | – | hourly, measurement campaign (2024-10-01 00:00 – 2025-09-30 23:00) |
| `residential_kwh` | kWh | residential category, already scaled to the 781 equivalent households of the community |
| `public_administration_kwh` | kWh | public buildings |
| `industrial_kwh` | kWh | industrial user |
| `street_lighting_kwh` | kWh | public street lighting |

Annual consumption in the case study: residential 2711.2 MWh, public administration 124.6 MWh, industrial
2397.6 MWh, street lighting 317.4 MWh (56 metered load points in total). The quarter-hourly data of the DSO are
aggregated to hourly values. The residential category is obtained from the 23 metered residential supply points
scaled by the ratio between the equivalent households of the community and the metered ones (781/23 ≈ 34), as
described in Section 3.1.

### `residential_sample.csv`

| Column | Unit | Description |
|---|---|---|
| `timestamp` | – | hourly, measurement campaign |
| `energy_kwh` | kWh | aggregate consumption of the 23 metered residential supply points |
| `energy_kwh_clustering` | kWh | same series in the export used for the clustering and the day-type classification |

The two columns come from two exports of the same meters and coincide everywhere except for two hours of
2024-10-27 (00:00 and 01:00), which are missing (stored as zero) in the export used for the clustering. Both columns
are kept so that every published number can be reproduced exactly; for a new dataset, the same series can be written
in both columns. The annual consumption of the sample is 79.8 MWh. The pipeline uses `energy_kwh` for the MLP
forecaster, for the EMS robustness test, and for the random-split benchmark, and `energy_kwh_clustering` for the
hierarchical clustering and for the random forest classifier.

## Synthetic data (`data/synthetic/`)

`python run_pipeline.py --make-synthetic` (or any run with `--data synthetic`) writes the two confidential files with
the format above, filled with synthetic profiles built from the public temperature, the calendar, and simple
behavioral patterns of each category, and scaled to the annual consumption of the case study
(`src/rec_pipeline/synthetic.py`). They make the whole pipeline executable without the confidential data, for testing
and for learning how it works. **The results obtained with them are not those of the paper.**
