# Frozen files

These files were written and recorded before the corresponding evaluation. They are included byte-identical; their
SHA-256 checksums are in `SHA256SUMS` (verify with `sha256sum -c protocols/SHA256SUMS` from the repository root).

| File | Content | Produced by |
|---|---|---|
| `../configs/stageBC_confirm.yaml` | configuration of the prespecified depth analysis (tests C1-C4) | fixed on participants 1-5 before the confirmatory run |
| `single_scale_kernel_parameters.json` | single-scale coherence kernel estimated from the ds004752 subdural grid | `analysis/kernel_single_scale_fit.py` |
| `single_scale_kernel_scalp_predictions.json` | scalp contrasts predicted from that kernel through the New York Head | `analysis/kernel_predict_scalp.py` |
| `two_component_model_and_scalp_predictions.json` | two-component model fitted on half of the grids, with its scalp predictions | `analysis/grid_modes_and_mixture_model.py fit` |

Evaluation status (Supporting Information S3):

- The single-scale prediction was recorded before the full ANPHY analysis, but after a preliminary analysis of nine
  ANPHY participants.
- The held-out grid evaluation of the two-component model was not blind.
- The ds003478 evaluation was blind: the predictions were recorded before any ds003478 measurement was computed.
