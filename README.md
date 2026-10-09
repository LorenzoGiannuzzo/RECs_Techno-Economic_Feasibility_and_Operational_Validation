# Techno-Economic Feasibility and Operational Assessment of a Renewable Energy Community

Code and public data of the paper

> L. Giannuzzo, F. Aretano, F.D. Minuto, A. Lanzini, *Techno-Economic Feasibility and Operational Assessment of a
> Renewable Energy Community in Southern Italy: From Simulation to Machine Learning-Driven Operation*, Sustainable
> Energy Technologies and Assessments (under review, manuscript SETA-D-26-06491).

The repository reproduces every numerical result, table, and figure of the paper: four BESS dispatch strategies under
the Italian CACER incentive scheme (no storage, rule-based self-consumption, price-driven arbitrage, and MILP
optimization), a 20-year discounted cash flow analysis with asset degradation, the allocation of the benefits among
the members, a machine-learning energy management system (hierarchical clustering, random forest day-type classifier,
and MLP load forecaster explained through SHAP), and an operational robustness test that measures how much of the
design-stage revenue survives when the dispatch is planned on forecasts and applied open-loop to the measured loads.

## Contents

1. [Repository structure](#1-repository-structure)
2. [Installation](#2-installation)
3. [Quick start with synthetic data](#3-quick-start-with-synthetic-data)
4. [Reproducing the results of the paper](#4-reproducing-the-results-of-the-paper)
5. [The pipeline step by step](#5-the-pipeline-step-by-step)
6. [Data: what is provided and what is missing](#6-data-what-is-provided-and-what-is-missing)
7. [Outputs](#7-outputs)
8. [Configuration](#8-configuration)
9. [Tests](#9-tests)
10. [Legacy thesis scripts](#10-legacy-thesis-scripts)
11. [Citation, license, and acknowledgments](#11-citation-license-and-acknowledgments)

## 1. Repository structure

```
.
├── run_pipeline.py              entry point (no installation required)
├── src/rec_pipeline/            the pipeline
│   ├── config.py                paths and every model parameter, with the section of the paper
│   ├── data.py                  loading and validation of the input files
│   ├── models.py                dispatch models (rule-based, MILP) and hourly economics
│   ├── design.py                step 1 – scenarios, financial analysis, sensitivity, benefit allocation
│   ├── clustering.py            step 2 – hierarchical clustering and random forest classifier
│   ├── forecasting.py           inputs and network of the MLP forecaster
│   ├── operation.py             step 3 – rolling-origin validation, SHAP, EMS robustness test
│   ├── extras.py                step 4 – complementary quantities quoted in the text
│   ├── figures.py               step 5 – figures of the paper
│   ├── synthetic.py             synthetic stand-in for the confidential data
│   └── __main__.py              command-line interface
├── data/
│   ├── README.md                description and format of every input file
│   ├── public/                  PV production, zonal prices, outdoor temperature (distributed)
│   └── confidential/            metered consumption of the community (NOT distributed)
├── results/                     published results (JSON)
├── figures/                     published figures (PNG and vector PDF, 500 dpi)
├── docs/
│   ├── methodology.md           code map: equations, tables, and figures of the paper
│   └── reproducibility.md       expected results, run times, determinism, troubleshooting
├── tests/                       unit tests
├── legacy/thesis_scripts/       original thesis scripts (archive, not needed to reproduce the paper)
├── requirements.txt             exact package versions used for the paper
├── pyproject.toml               package metadata and compatible versions
├── CITATION.cff
└── LICENSE
```

## 2. Installation

Python 3.10 or later is required (the published results were produced with Python 3.13). The MILP is solved with
CBC, which is bundled with PuLP, so no external solver needs to be installed.

```bash
git clone https://github.com/LorenzoGiannuzzo/RECs_Techno-Economic_Feasibility_and_Operational_Validation.git
cd RECs_Techno-Economic_Feasibility_and_Operational_Validation
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # exact versions used for the paper
```

Alternatively, `pip install -e .` installs the package with compatible versions and adds the `rec-pipeline` command,
which is equivalent to `python run_pipeline.py`.

## 3. Quick start with synthetic data

The metered consumption of the community is confidential (Section 6), so a fresh clone cannot reproduce the published
numbers. The whole pipeline can nonetheless be executed on a synthetic stand-in with the same format:

```bash
python run_pipeline.py --data synthetic
```

The command generates `data/synthetic/`, runs the five steps, and writes the outputs to `results_synthetic/` and
`figures_synthetic/` (about 4 minutes on a laptop). The synthetic profiles are plausible but are not the real ones, so
the results differ from those of the paper; they are meant for testing the code and for applying it to other
communities.

## 4. Reproducing the results of the paper

1. Place the two confidential files, `community_loads.csv` and `residential_sample.csv`, in `data/confidential/`
   (their format is described in [`data/README.md`](data/README.md)). Alternatively, set the environment variable
   `REC_CONFIDENTIAL_DIR` to the folder that contains them.
2. Run

   ```bash
   python run_pipeline.py
   ```

3. The results are written to `results/` and the figures to `figures/`, overwriting the published ones with identical
   values. The main numbers to check are listed in [`docs/reproducibility.md`](docs/reproducibility.md).

Single steps can be re-run with `--steps`, e.g. `python run_pipeline.py --steps figures` redraws the figures from the
existing outputs. The steps must follow the order design → clustering → operation → extras → figures, since each one
reads the outputs of the previous ones.

## 5. The pipeline step by step

A complete map between the code and the equations, tables, and figures of the paper is given in
[`docs/methodology.md`](docs/methodology.md). In short:

| Step | What it does | Paper |
|---|---|---|
| 0 – data | Validates the inputs, interpolates the hour skipped by the daylight-saving change, maps the measurement campaign (October 2024 – September 2025) onto the simulated year 2025 preserving the day of the week, and computes the incentive tariff | Sections 2.1, 3.1 |
| 1 – design | Simulates the four dispatch strategies, solves the annual MILP (52 560 variables, 8760 binary), computes NPV, IRR, and payback with PV and battery degradation over 20 years, the sensitivity to the BESS cost, the break-even cost, and the three-phase CACER allocation with the 55% threshold | Sections 2.1–2.3, 3.2, 4.1, 4.2; Tables 3–7; Figs. 3–6 |
| 2 – clustering | Clusters the normalized daily residential profiles (Ward linkage, k selected by Silhouette and Davies–Bouldin indices) and trains the random forest day-type classifier on temperature and calendar features | Sections 2.4, 4.3; Table 8; Fig. 7 |
| 3 – operation | Trains the day-ahead MLP forecaster with a rolling-origin validation (monthly retraining, January–September 2025 out-of-sample), compares it with naive predictors, computes the peak-timing error and the out-of-sample SHAP values, and runs the EMS robustness test (perfect-foresight MILP vs forecast-based plans applied open-loop) | Sections 2.4, 2.5, 4.3, 4.4; Tables 9–10; Figs. 8–10 |
| 4 – extras | Random-split benchmark of the forecaster, solve time of the 24-hour MILP, critical-week revenue gap, present value of the MILP revenue increment, price statistics | Sections 2.2, 4.3, 4.4, 5.3 |
| 5 – figures | Draws Figs. 2–10 as PNG and vector PDF | Figs. 2–10 |

## 6. Data: what is provided and what is missing

| File | Content | Status |
|---|---|---|
| `data/public/pv_production_2025.csv` | hourly PV production of the 1037 kWp / 1000 kW plant (PVsyst) | provided |
| `data/public/zonal_prices_south_2025.csv` | hourly day-ahead zonal prices of the South zone, 2025 (GME) | provided |
| `data/public/outdoor_temperature_hourly.csv` | hourly outdoor temperature over the measurement campaign | provided |
| `data/public/weather_daily.csv` | daily mean, maximum, and minimum temperature | provided |
| `data/confidential/community_loads.csv` | hourly consumption of the four user categories | **not distributed** |
| `data/confidential/residential_sample.csv` | hourly consumption of the 23 metered residential supply points | **not distributed** |

The two missing files contain the consumption of the members of the community, provided by the distribution system
operator under confidentiality, as stated in the Data Availability section of the paper. Their exact format (columns,
units, calendar, and conventions for missing hours) is documented in [`data/README.md`](data/README.md), so that they
can be replaced by the original data or by the data of another community, and `python run_pipeline.py --make-synthetic`
writes synthetic files with the same format.

## 7. Outputs

| File | Content |
|---|---|
| `results/results_design.json` | input totals, Table 3, scenarios (Tables 4–5), MILP size and checks, DCF with and without degradation (Table 6), sensitivity and break-even cost, yearly results, allocation (Table 7) |
| `results/results_clustering.json` | validity indices for k = 3–15, cluster sizes, RMSE and composition, classifier metrics, feature importance (Table 8) |
| `results/results_ops.json` | rolling-origin folds, out-of-sample metrics and benchmarks, peak-timing error, SHAP importance (Table 9), robustness test (Table 10) |
| `results/extra.json` | complementary quantities quoted in the text |
| `results/critical_week.json` | week shown in Fig. 10 |
| `figures/Fig01…Fig10` | figures of the paper (Fig. 1 is a drawing provided as an image) |

The hourly intermediate files (`results/*.npz`) are regenerated by every run and are not distributed, since they
contain or are derived from the metered loads.

## 8. Configuration

All the parameters of the paper (regulatory values, BESS size and efficiency, arbitrage thresholds, unit costs,
discount rate, degradation, allocation rules, forecaster settings, random seed) are defined once in
[`src/rec_pipeline/config.py`](src/rec_pipeline/config.py), each with the section of the paper in which it is
introduced. A sensitivity case only requires editing that file and re-running the pipeline.

```
python run_pipeline.py [--data {confidential,synthetic}] [--steps design,clustering,operation,extras,figures]
                       [--make-synthetic]
```

## 9. Tests

```bash
pip install pytest
python -m pytest
```

The tests generate synthetic data and check the validation of the inputs, the incentive tariff, the feasibility of
the MILP and of the rule-based strategies on short horizons (shared energy equal to the hourly minimum, no
simultaneous charging and discharging, SOC within its window), and the open-loop realization. They run automatically
on every push through GitHub Actions (`.github/workflows/tests.yml`).

## 10. Legacy thesis scripts

`legacy/thesis_scripts/` keeps the original scripts of the master's thesis from which the paper originates
(dispatch simulations, first clustering and forecasting experiments, GAN and MPC
explorations). They are an archive: they are not maintained, they read local data files that are not distributed,
and some of them contain the inconsistencies corrected during the revision of the paper (listed in
[`legacy/README.md`](legacy/README.md)). They are not needed to reproduce the paper.

## 11. Citation, license, and acknowledgments

If you use this code, please cite the paper (see [`CITATION.cff`](CITATION.cff)). The code is released under the MIT
License ([`LICENSE`](LICENSE)); the zonal prices are published by GME and remain subject to its terms of use.

L. Giannuzzo, F.D. Minuto, and A. Lanzini carried out this work within the project "Network 4 Energy Sustainable
Transition – NEST" (project code PE0000021, CUP E13C22001890001), funded under the National Recovery and Resilience
Plan (NRRP), Mission 4, Component 2, Investment 1.3, by the European Union – NextGenerationEU.

Contact: Lorenzo Giannuzzo, Energy Center Lab, Politecnico di Torino – lorenzo.giannuzzo@polito.it
