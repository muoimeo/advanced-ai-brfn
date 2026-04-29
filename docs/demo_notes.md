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
reports zero overlapping source-image IDs. External validation is still limited
because the custom-image set is small and banana-heavy.

Demo narrative
--------------

Use the API to show:

- 28-class prediction
- produce type and freshness status
- confidence
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
and risky high-confidence outputs for manual review where appropriate.

Privacy/data governance position: prediction logs should store metadata and
model outputs, not unnecessary personal data or raw images.
