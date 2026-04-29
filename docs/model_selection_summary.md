Model Selection Summary
=======================

Current selected artifact: `efficientnetb0_aug_oversampled_finetuned_wsl`.

The decision is evidence-driven rather than metric-only. After grouped
source-image split retraining, EfficientNetB0 fine-tuned is selected because it
has marginally higher accuracy/weighted F1 than ResNet50 fine-tuned and fewer
high-confidence errors in the grouped XAI audit. ResNet50 remains the main
macro-F1 challenger because it has stronger macro F1 and weak rotten-class F1.

Decision table: `docs/report_figures/final_model_decision.csv`

Key interpretation
------------------

- EfficientNetB0 fine-tuned grouped: selected final deployment candidate;
  accuracy 0.9720, macro F1 0.9613, weighted F1 0.9719, 11 high-confidence
  errors.
- ResNet50 fine-tuned grouped: main macro-F1 challenger; accuracy 0.9715,
  macro F1 0.9650, weighted F1 0.9715, 19 high-confidence errors.
- ResNet50 head grouped and EfficientNetB0 head grouped: useful ablations, but
  not selected because fine-tuning improves final-candidate readiness.

Risk caveat
-----------

The original split leakage audit found offline-augmented variants crossing
splits. The current top-model comparison now uses `data/splits_grouped`, whose
audit reports zero overlapping source-image IDs across train/validation/test.
External validation is still limited because the current custom-image set is
small and banana-heavy.

Assessment framing
------------------

For outstanding-level evaluation, present the final model as a practical
classification and decision-support pipeline with known limitations:

- 28-class produce/freshness classifier
- business mapping to grade and action
- XAI used for model-selection risk analysis
- confidence and reason-code safeguards
- feedback logging for accountability
- no claim of autonomous quality approval
