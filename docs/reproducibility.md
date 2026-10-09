# Reproducibility

## Expected results

With the confidential data in `data/confidential/` and the package versions of `requirements.txt`, the pipeline
reproduces the published values exactly. The main ones are:

| Quantity | Value | Paper |
|---|---|---|
| Annual PV production | 1666.0 MWh | Section 3.2 |
| Shared energy, no-BESS / MILP | 1212 / 1377 MWh/yr (+13.6%) | Table 4 |
| Gross revenue, no-BESS / MILP | 293 656 / 324 113 €/yr (+10.4%) | Table 4 |
| NPV, IRR, payback (no-BESS) | 1 083 758 €, 16.4%, 6.6 years | Table 6 |
| NPV, IRR, payback (MILP) | 955 707 €, 12.7%, 8.5 years | Table 6 |
| Break-even BESS cost of MILP vs no-BESS | 210 €/kWh | Section 4.2 |
| NPV with zonal prices −20% / +20% (no-BESS) | 667 263 / 1 500 254 € | Section 4.2 |
| NPV with zonal prices −20% / +20% (MILP) | 506 265 / 1 411 578 € | Section 4.2 |
| Surplus fraction θ and surplus quota | 0.340, 45 205 €/yr | Table 7 |
| Cluster sizes (days) | 224 / 92 / 49 | Section 4.3, Fig. 7 |
| Classifier accuracy, macro F1, end-to-end RMSE | 91.78%, 88.85%, 0.1113 | Section 4.3 |
| Classifier five-fold cross-validation accuracy (folds in chronological order within each archetype) | 89.0% | Section 4.3 |
| MLP out-of-sample R², RMSE, MAPE | 0.710, 67.0 kWh, 17.1% | Section 4.3 |
| Random-split R² (same model) | 0.783 | Section 4.3 |
| Naive R² (previous day / previous week) | 0.593 / 0.444 | Section 4.3 |
| Peak-timing error, MLP / previous-day persistence | 2.1 h / 2.3 h | Section 4.3 |
| Revenue gap, plans A / B / C | 0.54% / 0.84% / 0.93% | Table 10 |
| Injection-bounded hours with PV production (perfect foresight) | 62.1% | Sections 4.4, 5.4 |

The complete set of values is stored in the JSON files of `results/`, which can be compared with a new run, e.g.

```python
import json
old = json.load(open("results/results_design.json"))   # published
# ... run the pipeline into a copy of the repository, then compare the two dictionaries
```

## Determinism

- All the random components (MLP initialization and shuffling of the mini-batches, k-means background and sample selection of
  SHAP, random forest, train/test splits) use the seed defined in `config.SEED`.
- The MILP is solved to optimality by CBC; the optimal objective is unique, and the published dispatch is reproduced
  bit by bit with the versions of `requirements.txt`. With different versions of PuLP/CBC, alternative optimal
  dispatches with the same objective value may be returned, which can change the last digit of some hourly
  statistics without affecting the revenues.
- Different versions of scikit-learn or NumPy may change the training of the MLP and of the random forest in the
  last decimals.
- Only the run times stored in the results (`time_s`, `train_time_s`, `train_total_s`) change from run to run.

## Run time

On a laptop CPU, the full run takes about 8 minutes:

| Step | Time | Dominant cost |
|---|---|---|
| 1 – design | ~4.5 min | 61 annual MILP solutions (about 4 s each): the design scenario, the 20 years with degradation, and the 20 years of each of the two price levels of the price sensitivity |
| 2 – clustering | ~10 s | validity indices for 14 values of k |
| 3 – operation | ~1.5 min | nine MLP trainings, kernel SHAP, one MILP over 6552 hours and 819 daily MILPs |
| 4 – extras | ~10 s | one MLP training |
| 5 – figures | ~30 s | rendering at 500 dpi |

## Troubleshooting

| Message | Cause and solution |
|---|---|
| `missing input file .../community_loads.csv` | the confidential data are not in `data/confidential/`: copy them there, set `REC_CONFIDENTIAL_DIR`, or run with `--data synthetic` |
| `8760 consecutive hourly rows from ... are required` | the file does not follow the calendar of `data/README.md` (check the first timestamp and that every day has 24 rows, also on the daylight-saving days) |
| `empty values are not allowed` | store missing hours as `0`; they are interpolated by the pipeline |
| `AttributeError: ... LpVariable.dicts` | PuLP 3 is installed; install the version of `requirements.txt` (`pulp<3`) |
| figures with a different font | the figures use Liberation Sans or Arial when available and fall back to DejaVu Sans otherwise |
| `No such file ... .npz` when running `--steps figures` | the figures are drawn from the hourly outputs of steps 1–4, which are not distributed: run the whole pipeline once |
