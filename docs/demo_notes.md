BRFN freshness classifier demo notes
====================================

Current phase
-------------

This repository is in final interpretation and demo hardening, not initial
training. Completed evidence includes EDA/splits, baseline CNNs, MobileNetV2,
EfficientNetB0, ResNet50, fine-tuning, Grad-CAM audit, an XAI-corrected
EfficientNetB0 experiment, custom-image smoke testing, and FastAPI scaffolding.

Current model decision
----------------------

`efficientnetb0_aug_oversampled_finetuned_wsl` is the current selected artifact
after grouped source-image split retraining:

- test accuracy: 0.9720
- macro F1: 0.9613
- weighted F1: 0.9719
- high-confidence errors: 11

This is not a metric-only decision. ResNet50 fine-tuned remains the main
macro-F1 challenger:

- ResNet50 accuracy: 0.9715
- ResNet50 macro F1: 0.9650
- ResNet50 weighted F1: 0.9715
- ResNet50 high-confidence errors: 19

The report should explain this tradeoff: EfficientNetB0 is selected for
deployment-risk posture and slightly stronger overall/weighted performance,
while ResNet50 is stronger on macro F1 and weak rotten-class F1.

The selected model should therefore be presented as a decision-support model,
not an autonomous quality approval system.

Known evidence caveat
---------------------

The original split audit found offline-augmented variants of the same source
image crossing train/validation/test:

- overlapping source-image identities: 1361
- rows involved: 9769
- affected classes: 6

The current top-model comparison now uses `data/splits_grouped`, whose audit
reports zero overlapping source-image IDs.

External validation is now exported to:

- `docs/report_figures/custom_image_validation.csv`
- `docs/report_figures/custom_image_validation_summary.csv`

The current labeled custom-image subset is still small and banana-heavy. Extra
household photos without manifest labels are useful qualitative domain-shift
evidence, but they should not be counted as accuracy evidence until
`expected_class` is added to `docs/report_figures/custom_image_manifest.csv`.

Demo narrative
--------------

Use the API to show:

- 28-class prediction
- produce type and freshness status
- confidence
- top-1/top-2 prediction margin
- quality grade
- recommended action
- reason codes and manual-review flags
- model metadata from `/model-info`
- feedback logging for accountability

Use XAI figures to explain where the model focused, but state the limitation:
Grad-CAM is not a segmentation mask and does not prove perfect rotten-region
localization.

Deployment safeguards
---------------------

High-confidence errors are more dangerous than low-confidence uncertainty.
The demo should explicitly show that the system flags low-confidence predictions
and ambiguous predictions for manual review where appropriate.

Current inference safeguards:

- confidence below 0.75 triggers `LOW_CONFIDENCE` and manual review
- top-1/top-2 margin below 0.15 triggers `AMBIGUOUS_PREDICTION` and manual review
- rotten predictions expose `ROTTEN_PREDICTION`
- confidence at or above 0.95 exposes `HIGH_CONFIDENCE_PREDICTION`

Privacy/data governance position: prediction logs should store metadata and
model outputs, not unnecessary personal data or raw images.
