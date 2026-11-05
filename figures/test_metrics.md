| model | rule | threshold | accuracy | precision | recall | f1 | roc_auc |
|---|---|---|---|---|---|---|---|
| Baseline CNN | val-chosen t | 0.380 | 0.592 | 0.584 | 0.901 | 0.709 | 0.699 |
| Baseline CNN | default t=0.5 | 0.500 | 0.653 | 0.721 | 0.605 | 0.658 | 0.699 |
| MobileNetV2 (fine-tuned) | val-chosen t | 0.460 | 0.796 | 0.823 | 0.802 | 0.812 | 0.850 |
| MobileNetV2 (fine-tuned) | default t=0.5 | 0.500 | 0.810 | 0.853 | 0.790 | 0.821 | 0.850 |