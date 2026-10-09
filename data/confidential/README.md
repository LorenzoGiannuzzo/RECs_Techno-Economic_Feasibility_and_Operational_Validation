# Confidential data (not distributed)

This folder must contain the metered consumption of the community, which is confidential and is therefore not part
of the public repository (everything in this folder except this file is ignored by git):

- `community_loads.csv` – hourly consumption of the four user categories (October 2024 – September 2025);
- `residential_sample.csv` – hourly consumption of the 23 metered residential supply points.

The exact format of both files is described in [`data/README.md`](../README.md#confidential-data-dataconfidential-not-distributed).
Without these files, the pipeline can be run on synthetic data with:

```bash
python run_pipeline.py --data synthetic
```
