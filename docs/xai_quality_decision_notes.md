XAI and Quality Decision Notes
==============================

Purpose
-------

This project uses XAI in two different ways, and the report/demo should keep
them separate:

- Grad-CAM is attention evidence for model audit and model-selection risk.
- The quality grade/action is produced by the external rule-based layer in
  `src/quality/`.

EfficientNetB0 is not a supervised Grade A/B/C model. It predicts the 28-class
label, for example `Banana__Rotten` or `Apple__Healthy`. The rule layer then
maps that prediction plus image proxy features into grade, action, inventory
status, discount, reason codes, warnings, and manual-review flags.

What Grad-CAM Can Support
-------------------------

Use Grad-CAM outputs from `notebooks/07_xai_gradcam.ipynb`,
`notebooks/09_xai_corrected_before_after.ipynb`, and
`notebooks/custom_image_test.ipynb` to discuss:

- whether attention appears to overlap with the produce object;
- whether attention drifts to background, texture, edge, or lighting cues;
- high-confidence errors as deployment risk;
- why EfficientNetB0 fine-tuned is selected over ResNet50 fine-tuned despite
  ResNet50 having stronger macro F1.

What Grad-CAM Must Not Claim
----------------------------

Do not claim that Grad-CAM proves defect localization or segmentation-quality
rotten-area understanding. The dataset has image-level labels only, so there is
no pixel-level ground truth for defect regions.

Do not use Grad-CAM heatmaps as grading features in v1. They are report-facing
attention references, not runtime quality measurements.

Quality Layer Evidence
----------------------

Use `notebooks/10_quality_rule_validation.ipynb` and
`outputs/quality_rule_eval/` to show the rule layer behavior:

- grade distribution;
- manual-review rate;
- action distribution;
- weight sensitivity across `default_balanced`, `more_visual_colour`, and
  `safer_model_led`;
- percentage of rotten predictions mapped to `C/Review`;
- risky decisions, meaning rotten or uncertain predictions mapped to `A/B`.

Because there are no supervised grade labels, do not report grade accuracy.
Frame this as rule validation, decision-risk analysis, and deployment-readiness
evidence.

Current Custom-Image Interpretation
-----------------------------------

The current custom-image results show a useful deployment pattern:

- strong healthy predictions can receive Grade `A`;
- image-quality warnings can still recommend manual review because household
  images may have weak lighting, background clutter, or scale mismatch;
- rotten predictions are conservatively mapped to Grade `C`;
- weight sensitivity currently keeps risky `A/B` decisions at zero on the
  custom-image prediction set.

This supports the final position: the system is suitable as decision support
for producer/admin review, not as an autonomous quality authority.
