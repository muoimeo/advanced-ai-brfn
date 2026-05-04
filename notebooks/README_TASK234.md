# Tasks 2-4 Notebook Signposting

The existing root-level notebooks are the computer-vision evidence for Tasks 2-4.

They cover:

- EDA and grouped split work
- baseline CNN and augmentation experiments
- MobileNetV2, EfficientNetB0, and ResNet50 comparisons
- EfficientNetB0 fine-tuning
- Grad-CAM/XAI analysis
- custom image validation
- quality rule validation
- feedback logging and monitoring evidence for model interaction/refinement

Do not move these notebooks close to submission unless path references have been checked. The report should reference them as Tasks 2-4 evidence, separate from Task 1.

Task 3 monitoring evidence is implemented outside the notebooks in:

```text
src/monitoring/feedback_monitoring.py
GET /monitoring/feedback-summary
outputs/final_evaluation/feedback_monitoring_summary.json
outputs/final_evaluation/feedback_accuracy_over_time.csv
```

This monitoring layer joins prediction logs and feedback logs by `prediction_id`.
It reports a human-feedback accuracy proxy over time, not controlled test-set
accuracy.
