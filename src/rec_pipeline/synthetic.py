"""Synthetic stand-in for the confidential metered data.

The metered consumption of the community cannot be shared (see data/README.md). This module writes two files with
exactly the same format as the confidential ones, filled with plausible synthetic profiles, so that the whole pipeline
can be executed and tested without the original data. The synthetic profiles are built from the public outdoor
temperature, from the calendar, and from simple behavioral patterns of each user category, and they are scaled to the
annual consumption reported in the paper; the results obtained with them are NOT those of the paper.
"""
import numpy as np
import pandas as pd

from . import config as C

#Lorenzo Giannuzzo: annual consumption of the four categories reported in the paper [MWh] and residential sample size
ANNUAL_MWH = {"residential_kwh": 2711.2, "public_administration_kwh": 124.6, "industrial_kwh": 2397.6,
              "street_lighting_kwh": 317.4}
SAMPLE_MWH = 79.8
LATITUDE = 39.8


def _night_mask(ts):
    """Hours between sunset and sunrise at the latitude of the case study (approximate solar geometry)."""
    doy = ts.dayofyear.values
    decl = np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    ha = np.degrees(np.arccos(-np.tan(np.radians(LATITUDE)) * np.tan(decl)))
    sunrise, sunset = 12 - ha / 15, 12 + ha / 15
    h = ts.hour.values + 0.5
    return (h < sunrise) | (h > sunset)


def _scale(x, mwh):
    return x * (mwh * 1e3 / x.sum())


def generate(out_dir=C.SYNTHETIC_DIR, seed=C.SEED):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(C.MEAS_YEAR_START, periods=C.HOURS, freq="h")
    temp = pd.read_csv(C.PUBLIC_DIR / "outdoor_temperature_hourly.csv").temperature_c.values
    h = ts.hour.values; dow = ts.dayofweek.values
    off = (dow >= 5) | ts.normalize().isin(pd.to_datetime(C.HOLIDAYS))
    hh = np.arange(24)

    #Lorenzo Giannuzzo: residential load: morning and evening peaks, later and flatter on non-working days, with
    # heating and cooling components driven by the outdoor temperature and autocorrelated noise
    work = 0.35 + 0.25 * np.exp(-0.5 * ((hh - 7.5) / 1.2) ** 2) + 0.55 * np.exp(-0.5 * ((hh - 20.5) / 1.8) ** 2) \
        + 0.15 * np.exp(-0.5 * ((hh - 13.5) / 1.5) ** 2)
    rest = 0.40 + 0.35 * np.exp(-0.5 * ((hh - 12.5) / 1.8) ** 2) + 0.45 * np.exp(-0.5 * ((hh - 20.5) / 2.0) ** 2)
    base = np.where(off, rest[h], work[h])
    heat = 0.025 * np.maximum(0, 14 - temp) * ((h >= 6) & (h <= 23))
    cool = 0.045 * np.maximum(0, temp - 25) * ((h >= 12) | (h <= 1))
    noise = np.empty(C.HOURS); noise[0] = 0.0
    eps = rng.normal(0, 0.08, C.HOURS)
    for t in range(1, C.HOURS):
        noise[t] = 0.7 * noise[t - 1] + eps[t]
    sample = _scale(np.maximum(0.05, (base + heat + cool) * (1 + noise)), SAMPLE_MWH)
    residential = _scale(sample * C.K_SCALE * (1 + rng.normal(0, 0.01, C.HOURS)), ANNUAL_MWH["residential_kwh"])

    #Lorenzo Giannuzzo: public administration: office hours on working days and a low base load otherwise
    pa = np.where(off, 0.25, np.where((h >= 8) & (h < 18), 1.0, 0.3)) * (1 + 0.15 * np.maximum(0, temp - 26) / 5)
    pa = _scale(pa * (1 + rng.normal(0, 0.05, C.HOURS)), ANNUAL_MWH["public_administration_kwh"])

    #Lorenzo Giannuzzo: industrial user: withdrawal concentrated in the evening and night hours of working days, a
    # contraction in the central hours, reduced levels on weekends, and a two-week shutdown in August
    ind = np.where((h >= 17) | (h < 6), 1.0, np.where((h >= 10) & (h < 15), 0.35, 0.6))
    ind = np.where(dow == 6, 0.25, np.where(dow == 5, 0.6 * ind, ind))
    ind = np.where((ts.month == 8) & (ts.day >= 8) & (ts.day <= 21), 0.15, ind)
    ind = _scale(ind * (1 + rng.normal(0, 0.06, C.HOURS)), ANNUAL_MWH["industrial_kwh"])

    #Lorenzo Giannuzzo: street lighting: constant power between sunset and sunrise
    sl = _scale(np.where(_night_mask(ts), 1.0, 0.0) * (1 + rng.normal(0, 0.01, C.HOURS)),
                ANNUAL_MWH["street_lighting_kwh"])

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = ts.strftime("%Y-%m-%d %H:%M")
    pd.DataFrame({"timestamp": stamp, "residential_kwh": residential.round(3),
                  "public_administration_kwh": pa.round(3), "industrial_kwh": ind.round(3),
                  "street_lighting_kwh": sl.round(3)}).to_csv(out_dir / "community_loads.csv", index=False)
    pd.DataFrame({"timestamp": stamp, "energy_kwh": sample.round(3),
                  "energy_kwh_clustering": sample.round(3)}).to_csv(out_dir / "residential_sample.csv", index=False)
    return out_dir
