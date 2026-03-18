# Excluded Experiments

This archive note records experiments that are intentionally excluded from the main conference-ready package.

## Reason for Exclusion

The final study is framed around classical machine learning only. Preliminary neural-network experiments were explored during development, but they are not part of the main methodology because:

- tree and boosting models performed better or comparably under the strict spatial holdout
- classical models are easier to defend for an interpretable conference submission
- the final repository is centered on leakage-aware classical ML rather than neural architectures

## Archived Topics

- DNN experiments
- ResNet experiments
- PyTorch / Lightning training workflows
- neural-network SHAP outputs and related figures
- legacy benchmark scripts and intermediate workspace folders from the development phase

## Archived Layout

- `scripts/` contains the legacy DNN, ResNet, and early benchmarking scripts.
- `artifacts/` contains development-stage figures, reports, checkpoints, and handoff notes.
- `legacy_workspace/` contains superseded processed folders and the old root-level dataset copy.

These materials are retained for internal reference only. They are excluded from the main paper, README workflow, and final results narrative.
