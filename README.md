# Aperiodic exponent estimates across spatial derivations: analysis code

Analysis code for the manuscript *Aperiodic exponent estimates across spatial derivations in human intracranial and
scalp recordings*. It measures how the aperiodic (1/f-like) spectral exponent changes with the spatial derivation of the same
recordings. All contrasts use zero-sum derivations, so a reference signal common to all electrodes cancels:

- bipolar contact separation in stereo-EEG;
- graph spatial modes of subdural grids;
- average reference versus surface Laplacian at the scalp.

All datasets are public (see [Data](#data)). The repository contains no data and no results.

## Repository layout

```
analysis/       analysis scripts, one per step (see "Analyses and manuscript items")
src/apex/       shared library: data access, spectral estimators, derivations, statistics
configs/        analysis configurations for the prespecified depth analysis
protocols/      frozen model parameters and predictions, with SHA-256 checksums
verification/   numerical checks of the theory in Supporting Information S1 (no data needed)
tests/          unit tests on synthetic signals (no data needed)
```

## Installation

The analyses were run with Python 3.10 on Linux (32 CPU cores, 125 GB RAM; one GPU for the head-model computations).

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions used for the reported results. The scripts add `src/` to the import path
themselves. Alternatively, `pip install -e .` installs the `apex` library.

Quick checks that need no data:

```
python tests/test_core.py
python tests/test_specparam_synthetic.py
python verification/verify_theory.py
```

## Data

Download each dataset and place it under one data folder, given by `APEX_DATA`:

| Dataset | Source | Folder under `$APEX_DATA` |
|---|---|---|
| Simultaneous depth and scalp EEG (ds004752, v1.0.1) | OpenNeuro, doi:10.18112/openneuro.ds004752.v1.0.1 | `ds004752/` |
| HUP intracranial EEG (ds004100, v1.1.3) | OpenNeuro, doi:10.18112/openneuro.ds004100.v1.1.3 | `HUP_ds004100/` |
| ANPHY-Sleep | Open Science Framework, doi:10.17605/OSF.IO/R26FH | `ANPHY_Sleep/` (the `EPCTL*.zip` files and `Co-registered average positions.pos`) |
| Resting EEG (ds003478, v1.1.0) | OpenNeuro, doi:10.18112/openneuro.ds003478.v1.1.0 | `ds003478/` |
| MODMA 128-channel resting EEG | MODMA repository (registration required) | `MDD Datasets/EEG_128channels_resting_lanzhou_2015/` (`.mat` recordings) and `MODMA FIF Format 128 channels cleaned/` (FIF copies; only channel names and positions are read from these in the main analysis) |
| New York Head model | Huang et al. (2016), NeuroImage 140:150-162 | `nyhead/sa_nyhead.mat`, or set `APEX_NYHEAD` |

Set the locations before running anything:

```
export APEX_DATA=/path/to/data          # folder containing the datasets above
export APEX_WORK=/path/to/work          # intermediate caches go to $APEX_WORK/cache (tens of GB)
export APEX_NYHEAD=/path/to/sa_nyhead.mat   # optional; defaults to $APEX_DATA/nyhead/sa_nyhead.mat
```

Results are written to `results/` under the current working directory; run all commands from the repository root.

## Running the analyses

The steps below are listed in dependency order. Steps that compute spectra over all channel pairs are the
computationally heavy ones; use `--jobs` to set the number of parallel workers.

**1. Prespecified depth analysis (ds004752)**

```
python analysis/depth_contact_inventory.py --out results/inventory
python analysis/depth_prespecified_spectra.py --config configs/stageB_dev.yaml --jobs 12
python analysis/depth_prespecified_spectra.py --config configs/stageBC_confirm.yaml --jobs 12
python analysis/depth_prespecified_tests.py --config configs/stageB_dev.yaml --out results/stageBC_dev
python analysis/depth_prespecified_tests.py --config configs/stageBC_confirm.yaml --out results/stageBC_confirm
```

**2. Independent replication (HUP)**

```
python analysis/depth_replication_hup.py --out results/hup_c1 --jobs 12
```

**3. Depth robustness, anatomy, estimators and peak removal**

```
python analysis/depth_robustness_spectra.py --jobs 20
python analysis/depth_robustness_tests.py --out results/robust
python analysis/depth_anatomy_pathology.py --out results/straddle --jobs 12
python analysis/depth_anatomy_classifiable.py
python analysis/depth_distance_profile.py
python analysis/depth_welch_spectra.py --jobs 12
python analysis/depth_estimator_peak_tests.py --jobs 16
```

**4. Subdural grids**

```
python analysis/grid_modes_and_mixture_model.py measure --jobs 10
python analysis/grid_robustness.py measure --jobs 8
python analysis/grid_robustness.py analyze --jobs 16
```

**5. Scalp**

```
python analysis/scalp_anphy_segments_ica.py --out results/anphy_v2 --jobs 8
python analysis/scalp_ds003478_eye_state_ocular.py --out results/ds3478_followup --jobs 16
python analysis/scalp_derivations.py measure --dataset anphy --jobs 12
python analysis/scalp_derivations.py measure --dataset ds003478 --jobs 12
python analysis/scalp_derivations.py measure --dataset modma --jobs 12
python analysis/scalp_derivations.py analyze --jobs 16
python analysis/scalp_anphy_ecg_regression.py measure --jobs 8
python analysis/scalp_anphy_ecg_regression.py analyze --jobs 16
python analysis/scalp_preprocessing_variants.py --dataset ds003478 --out results/arbitration --jobs 16
python analysis/scalp_preprocessing_variants.py --dataset anphy --out results/arbitration --jobs 16
```

**6. Spectral model, fitting range and slope definition; further controls**

```
python analysis/spectral_specification_checks.py models --jobs 20
python analysis/spectral_specification_checks.py measure --jobs 20
python analysis/spectral_specification_checks.py eog
python analysis/spectral_specification_checks.py interp
python analysis/spectral_specification_checks.py grid
python analysis/spectral_specification_checks.py common --jobs 24
python analysis/specification_contrasts.py
```

**7. Mechanistic models**

```
python analysis/kernel_single_scale_fit.py --out results/grid7a
python analysis/kernel_predict_scalp.py
python analysis/kernel_compare_scalp.py
python analysis/grid_modes_and_mixture_model.py fit
python analysis/mixture_compare_scalp.py
```

`kernel_head_model.py` provides the New York Head lead-field and derivation-gain functions used by
`kernel_predict_scalp.py`.

**8. Statistical inventory (Data S1)**

```
python analysis/statistical_inventory.py
```

## Analyses and manuscript items

| Manuscript item | Script(s) |
|---|---|
| Table 2 (prespecified tests C1-C4) | `depth_prespecified_spectra.py`, `depth_prespecified_tests.py` |
| HUP replication (Results 3.1; Figure 2a) | `depth_replication_hup.py` |
| Depth robustness (Table S1) | `depth_robustness_spectra.py`, `depth_robustness_tests.py`, `depth_anatomy_pathology.py` |
| Contiguous and classifiable pairs (Results 3.1) | `depth_anatomy_pathology.py`, `depth_anatomy_classifiable.py` |
| Distance profile (Figure 2b; S2) | `depth_distance_profile.py` |
| Specparam, peak removal, adjacent and non-adjacent pairs (Figure 2c,d) | `depth_welch_spectra.py`, `depth_estimator_peak_tests.py` |
| Subdural grids (Results 3.2; Figure 3; Table S2) | `grid_modes_and_mixture_model.py` (measure), `grid_robustness.py` |
| Scalp P-2, pair slope, sub-bands, eye state (Results 3.3; Figure 4; Table S3) | `scalp_derivations.py`, `scalp_ds003478_eye_state_ocular.py`, `scalp_anphy_segments_ica.py` |
| ICA, EOG and ECG controls; interpolated channels (S4) | `scalp_anphy_segments_ica.py`, `scalp_ds003478_eye_state_ocular.py`, `scalp_anphy_ecg_regression.py`, `spectral_specification_checks.py` (eog, interp) |
| Spectral model, fitting range, slope definition (Results 3.4; Figure 5; Tables S4-S6) | `spectral_specification_checks.py` (models, common), `specification_contrasts.py` |
| Graph-mode zero-sum check (Methods 2.3) | `spectral_specification_checks.py` (grid) |
| Mechanistic models (Results 3.5; S3) | `kernel_single_scale_fit.py`, `kernel_head_model.py`, `kernel_predict_scalp.py`, `kernel_compare_scalp.py`, `grid_modes_and_mixture_model.py` (fit), `mixture_compare_scalp.py` |
| Multiplicity inventory (Data S1) | `statistical_inventory.py` |
| Theory checks (S1) | `verification/verify_theory.py` |

## Prespecified analyses and frozen files

- `configs/stageBC_confirm.yaml` is the configuration fixed on the development participants (1-5) before any
  confirmatory analysis; the four prespecified tests were run once with it on participants 6-15. It is included
  byte-identical to the frozen file (its comments refer to the internal analysis record), and its SHA-256 is listed in
  `protocols/SHA256SUMS`.
- The HUP replication protocol, fixed before any HUP exponent was computed, is stated at the top of
  `analysis/depth_replication_hup.py`.
- `protocols/` holds the frozen parameters and predictions of the two mechanistic models, byte-identical to the files
  recorded before evaluation (see `protocols/README.md`).

For release, the analysis scripts were renamed, data locations were made configurable through `apex.paths`, and
module descriptions were rewritten. Computational code is unchanged. Rerunning the released scripts on the original
intermediate files reproduces the existing outputs.
