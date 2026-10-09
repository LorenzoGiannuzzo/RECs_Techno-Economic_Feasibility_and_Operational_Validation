# Legacy thesis scripts (archive)

This folder keeps the original scripts of the master's thesis from which the paper originates, for transparency and
traceability. **They are not needed to reproduce the paper**, which is entirely reproduced by `src/rec_pipeline/`, and
they are not maintained.

## Status

- The scripts read local data files (spreadsheets, CSV, and MATLAB files) that contain or are derived from the
  confidential metered consumption of the community; these files are not distributed, so the scripts cannot be run
  from a fresh clone. Local absolute paths and the names of the local files have been replaced by neutral
  placeholders (`local_data/...`, `*_cs.csv`).
- Comments and variable names are partly in Italian, as in the original work.
- `simulation/sensitivity.py` was truncated in the original archive; it now ends with a note at the truncation point.

## Inconsistencies corrected in the published pipeline

The revision of the paper compared these scripts line by line with the manuscript; the published pipeline corrects
the following points, which are also listed in the response to the reviewers:

| Script | Issue | Correction in `src/rec_pipeline/` |
|---|---|---|
| `simulation/MILP_1.py` | the PVsyst export is read with `skiprows=10`, which delays the PV series of the MILP scenario by one hour with respect to the other scenarios | the corrected PV series is stored in `data/public/pv_production_2025.csv` and used by all the strategies |
| simulation scripts | the description of the incentive in the original manuscript (RID on the residual injection, TIP fixed component of 70 €/MWh) did not match the implementation | RID on the whole injection, TIP = min(100, 60 + max(0, 180 − P_zone)) €/MWh, ARERA valorization of 2025 (11.89 €/MWh) |
| `residential_clustering/res_2*.py` | cyclic encodings with periods of 23 h and 6 days, scaling parameters fitted on the whole dataset, random 70/30 split | periods of 24 h and 7 days, scaling on the training window only, rolling-origin validation, day-ahead inputs |
| load files of the four categories | measured year stored on the 2025 calendar, daylight-saving hour stored as zero | chronological series, interpolation of the skipped hour, explicit mapping onto the simulated year |

## Contents

| Folder | Scripts | Content |
|---|---|---|
| `simulation/` | `MILP.py`, `MILP_0.py`, `MILP_1.py` | first versions of the PV–BESS dispatch MILP and of the scenario simulations |
| | `MILP_ANN.py`, `MILP_GAN.py`, `GAN.py` | MILP driven by forecast or GAN-generated residential profiles (exploratory, not in the paper) |
| | `simulatore_reale.py`, `simulatore_mpc_avanzato.py` | open-loop and MPC simulations of the forecast-based plan (the MPC is discussed only qualitatively in the paper) |
| | `flussidicassa+.py`, `flussidicassasens.py`, `sensitivity.py`, `fig8_sensitivity_EN.py` | first versions of the cash-flow analysis and of the BESS cost sensitivity |
| | `analisi_guadagni_avanzata.py`, `heatmap.py`, `produc_FV.py` | revenue analysis and figures of the thesis |
| `residential_clustering/` | `res_1*.py`, `residenziali_case_study.py` | hierarchical clustering of the residential profiles and random forest classifier (ported to `clustering.py`) |
| | `res_2*.py` | first version of the MLP forecaster |
| `aggregated_load/` | `agg_1.py`, `aggregato_case_study.py` | clustering of the aggregated community load (exploratory) |
