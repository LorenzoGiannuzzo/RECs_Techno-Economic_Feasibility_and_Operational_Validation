"""Loading and validation of the input data (formats documented in data/README.md)."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

PUBLIC_FILES = {
    "pv": ("pv_production_2025.csv", ["timestamp", "pv_kwh"], C.SIM_YEAR_START),
    "prices": ("zonal_prices_south_2025.csv", ["timestamp", "price_eur_mwh"], C.SIM_YEAR_START),
    "temperature": ("outdoor_temperature_hourly.csv", ["timestamp", "temperature_c"], C.MEAS_YEAR_START),
}
WEATHER_DAILY = ("weather_daily.csv", ["date", "t_avg_c", "t_max_c", "t_min_c"])
LOAD_FILES = {
    "loads": ("community_loads.csv", ["timestamp", "residential_kwh", "public_administration_kwh", "industrial_kwh",
                                      "street_lighting_kwh"], C.MEAS_YEAR_START),
    "sample": ("residential_sample.csv", ["timestamp", "energy_kwh", "energy_kwh_clustering"], C.MEAS_YEAR_START),
}
CATEGORIES = {"Residential": "residential_kwh", "Public administration": "public_administration_kwh",
              "Industrial": "industrial_kwh", "Street lighting": "street_lighting_kwh"}
NON_BUSINESS = ["Residential", "Public administration", "Street lighting"]


class DataError(RuntimeError):
    pass


def _read_hourly(folder, name, columns, start):
    path = Path(folder) / name
    if not path.exists():
        raise DataError(f"missing input file {path}. See data/README.md for the expected format; the pipeline can be "
                        "run on synthetic data with: python run_pipeline.py --data synthetic")
    #Lorenzo Giannuzzo: round-trip parsing, so that the values are read back exactly as written
    df = pd.read_csv(path, float_precision="round_trip")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataError(f"{path}: missing columns {missing} (expected {columns})")
    expected = pd.date_range(start, periods=C.HOURS, freq="h")
    ts = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M")
    if len(df) != C.HOURS or not (ts.values == expected.values).all():
        raise DataError(f"{path}: {C.HOURS} consecutive hourly rows from {start} are required "
                        f"(found {len(df)} rows from {ts.iloc[0] if len(ts) else 'n/a'})")
    values = df[columns[1:]]
    if values.isna().any().any():
        raise DataError(f"{path}: empty values are not allowed (store missing hours as 0)")
    return df


@dataclass
class Inputs:
    """All the series used by the pipeline, on the calendars described in data/README.md."""
    ts_sim: pd.DatetimeIndex
    ts_meas: pd.DatetimeIndex
    pv: np.ndarray
    pz: np.ndarray
    tip: np.ndarray
    temperature: np.ndarray
    weather_daily: pd.DataFrame
    cats_chrono: dict
    cats: dict
    dem: np.ndarray
    res_sample: np.ndarray
    res_sample_clustering: np.ndarray


def tip_tariff(pz):
    """Incentive tariff of the CACER Decree for plants above 600 kW in southern Italy (Eq. 2)."""
    return np.minimum(C.TIP_CAP, C.TIP_FIXED + np.maximum(0.0, C.TIP_PRICE_REF - pz))


def load_inputs(loads_dir):
    pub = {k: _read_hourly(C.PUBLIC_DIR, *v) for k, v in PUBLIC_FILES.items()}
    met = {k: _read_hourly(loads_dir, *v) for k, v in LOAD_FILES.items()}
    wd = pd.read_csv(C.PUBLIC_DIR / WEATHER_DAILY[0], parse_dates=["date"], float_precision="round_trip")
    if list(wd.columns) != WEATHER_DAILY[1] or len(wd) != 365:
        raise DataError(f"{WEATHER_DAILY[0]}: 365 daily rows with columns {WEATHER_DAILY[1]} are required")
    ts_sim = pd.date_range(C.SIM_YEAR_START, periods=C.HOURS, freq="h")
    ts_meas = pd.date_range(C.MEAS_YEAR_START, periods=C.HOURS, freq="h")
    pz = pub["prices"].price_eur_mwh.values.astype(float)

    #Lorenzo Giannuzzo: the hour skipped by the daylight-saving change is stored as zero by the meters and is filled by
    # linear interpolation, as stated in the data pre-processing description
    dst = np.where(ts_meas.strftime("%Y-%m-%d %H:%M") == C.DST_SKIPPED_HOUR)[0]
    cats_chrono = {}
    for name, col in CATEGORIES.items():
        s = met["loads"][col].astype(float).copy()
        s.iloc[[i for i in dst if s.iloc[i] == 0]] = np.nan
        cats_chrono[name] = s.interpolate().values

    #Lorenzo Giannuzzo: the measured year is mapped onto the simulated year 2025: January-September are used as
    # measured, October-December 2024 are moved to October-December 2025 preserving the day of the week (2024-10-02
    # and 2025-10-01 are both Wednesdays), and the last day of 2025 reuses 2024-12-31
    cats = {k: np.r_[v[2208:], v[24:2208], v[2184:2208]] for k, v in cats_chrono.items()}
    return Inputs(ts_sim=ts_sim, ts_meas=ts_meas, pv=pub["pv"].pv_kwh.values.astype(float), pz=pz,
                  tip=tip_tariff(pz), temperature=pub["temperature"].temperature_c.values.astype(float),
                  weather_daily=wd, cats_chrono=cats_chrono, cats=cats, dem=sum(cats.values()),
                  res_sample=met["sample"].energy_kwh.values.astype(float),
                  res_sample_clustering=met["sample"].energy_kwh_clustering.values.astype(float))
