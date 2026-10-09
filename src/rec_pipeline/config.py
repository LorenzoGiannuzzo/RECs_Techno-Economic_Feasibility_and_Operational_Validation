"""Paths and model parameters of the pipeline.

Every numerical assumption of the paper is defined here once, together with the section of the paper in which it is
introduced, so that a sensitivity run only requires changing this file.
"""
import os
from dataclasses import dataclass
from pathlib import Path

#Lorenzo Giannuzzo: repository folders
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PUBLIC_DIR = DATA_DIR / "public"
CONFIDENTIAL_DIR = DATA_DIR / "confidential"
SYNTHETIC_DIR = DATA_DIR / "synthetic"


@dataclass(frozen=True)
class RunPaths:
    """Input folder of the metered data and output folders of one run."""
    mode: str
    loads_dir: Path
    results_dir: Path
    figures_dir: Path

    def ensure(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        return self


def run_paths(mode="confidential"):
    """Folders of a run on the metered data ('confidential') or on the synthetic data ('synthetic').

    The synthetic run writes to separate folders, so that the published results are never overwritten. The folder of
    the metered data can be moved outside the repository through the REC_CONFIDENTIAL_DIR environment variable.
    """
    if mode == "confidential":
        loads = Path(os.environ.get("REC_CONFIDENTIAL_DIR", CONFIDENTIAL_DIR))
        return RunPaths(mode, loads, ROOT / "results", ROOT / "figures")
    if mode == "synthetic":
        return RunPaths(mode, SYNTHETIC_DIR, ROOT / "results_synthetic", ROOT / "figures_synthetic")
    raise ValueError(f"unknown data mode {mode!r}: use 'confidential' or 'synthetic'")


#Lorenzo Giannuzzo: calendars. The PV production and the zonal prices refer to the simulated year 2025, whereas the
# metered loads cover the measurement campaign from October 2024 to September 2025 (Section 3.1); the hour skipped by
# the daylight-saving change is stored as zero by the meters and is interpolated (Section 2.4), and the national
# holidays are treated as non-working days by the forecaster (Section 2.4)
SIM_YEAR_START = "2025-01-01 00:00"
MEAS_YEAR_START = "2024-10-01 00:00"
HOURS = 8760
DST_SKIPPED_HOUR = "2025-03-30 02:00"
HOLIDAYS = ["2024-11-01", "2024-12-08", "2024-12-25", "2024-12-26", "2025-01-01", "2025-01-06", "2025-04-20",
            "2025-04-21", "2025-04-25", "2025-05-01", "2025-06-02", "2025-08-15"]

#Lorenzo Giannuzzo: Italian regulatory framework (Section 2.1), in €/MWh. The incentive tariff of plants above 600 kW
# is TIP = min(TIP_CAP, TIP_FIXED + max(0, TIP_PRICE_REF - Pz)), with no geographic correction in southern Italy, and
# the ARERA valorization is the 2025 TRASe value
ARERA_VALORIZATION = 11.89
TIP_FIXED = 60.0
TIP_CAP = 100.0
TIP_PRICE_REF = 180.0

#Lorenzo Giannuzzo: BESS and dispatch strategies (Section 2.2): capacity [kWh], converter rating [kW], SOC window,
# one-way efficiency (95% round trip), and discharge thresholds of the arbitrage strategy [€/MWh]
E_NOM = 760.0
P_MAX = 380.0
SOC_MIN, SOC_MAX, SOC0 = 0.10, 1.00, 0.50
ETA = 0.95 ** 0.5
ARB_PRICE_ANY_HOUR = 125.0
ARB_PRICE_EVENING = 85.0
ARB_EVENING_HOUR = 18

#Lorenzo Giannuzzo: financial analysis and benefit allocation (Section 2.3): DC capacity [kWp], unit costs [€/kWp and
# €/kWh], sensitivity range of the BESS cost [€/kWh], OPEX [% of CAPEX per year], discount rate, horizon [years],
# PV degradation [per year], rated cycle life and end-of-life state of health of the battery, share of the incentive
# retained by the REC entity, threshold of the CACER Decree [%], and equivalent households of the community
P_DC = 1037.0
C_PV = 800.0
C_BESS = 350.0
SENS_COSTS = [150, 250, 350, 450, 550]
OPEX_PCT = 1.5
RATE = 0.04
N_YEARS = 20
PV_DEG = 0.005
CYCLE_LIFE, SOH_EOL = 8000.0, 0.80
ALPHA = 0.10
RHO_STAR = 55.0
N_HOUSEHOLDS = 781

#Lorenzo Giannuzzo: machine-learning EMS (Section 2.4): scaling factor from the 23 metered residential PODs to the
# community (Section 3.1), number of clusters and range explored by the validity indices, and random seed
K_SCALE = 33.96
CLUSTER_K = 3
CLUSTER_K_RANGE = range(3, 16)
SEED = 42
