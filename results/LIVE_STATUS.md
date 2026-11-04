# Live status

Last updated 2026-11-04 19:09 (written automatically by scripts/live_status.py).

## Pretrained cascade (EfficientNet-B0 under every stage)

- **cascade**: accuracy 0.8317, macro-F1 0.6998, kappa 0.7643
- **flat**: accuracy 0.8260, macro-F1 0.7078, kappa 0.7609
- cascade stage errors: {'stage1_missed_defects': 75, 'stage1_false_alarms': 52, 'stage2_wrong_family': 272, 'stage3_wrong_subtype': 106, 'total_errors': 505, 'early_exit_rate': 0.5076666666666667}

## 10-fold cross-validation of the pretrained cascade

- cv_streamA.log: `[triage] ep   3 loss 0.2067 val_acc 0.9195 val_f1 0.9194 (25.3s)`
- cv_streamB.log: `[triage] ep   3 loss 0.2059 val_acc 0.9150 val_f1 0.9150 (25.2s)`
