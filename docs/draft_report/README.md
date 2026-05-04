# Draft LaTeX Technical Report

This folder contains a rewritten Advanced AI technical report draft aligned with the implemented repository.

Files:

- `main.tex` - report body, tables and architecture figure.
- `main.txt` - same report content kept as LaTeX-style text for easier editing when `.tex` tooling is unstable.
- `references.bib` - Harvard-style bibliography data used by `natbib`.
- `README.md` - build notes.

Compile from this folder:

```powershell
cd docs\draft_report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The draft intentionally avoids claiming:

- production recommender accuracy from synthetic Task 1 data;
- supervised Grade A/B/C learning by the CNN;
- rotten-region localisation, bounding boxes or segmentation;
- automatic retraining from feedback logs.

Evidence sources used in the draft:

- `outputs/task1_recommender/task1_summary.json`
- `outputs/task1_recommender/recommender_metrics.csv`
- `outputs/task1_recommender/discovery_metrics.csv`
- `outputs/final_evaluation/final_model_metrics.json`
- `outputs/final_evaluation/model_comparison_summary.csv`
- `outputs/final_evaluation/all_cv_model_confidence_audit_summary.csv`
- `outputs/final_evaluation/feedback_monitoring_summary.json`
- `outputs/final_evaluation/feedback_accuracy_over_time.csv`
- `outputs/final_evaluation/cost_sensitive_threshold_results.csv`
- `outputs/final_evaluation/cost_sensitive_threshold_summary.json`
- `outputs/final_evaluation/quality_score_ablation.csv`
- `outputs/task1_recommender/producer_fair_reranking_alpha_study.csv`
- `outputs/task1_recommender/producer_fair_reranking_summary.json`
- `models/model_metadata.json`

Monitoring wording used in the draft:

- `/feedback` records accepted AI recommendations and producer/admin overrides.
- `/monitoring/feedback-summary` reports class/grade accuracy proxies over time.
- These proxies are interaction monitoring evidence, not controlled test-set accuracy.

Producer forecast wording used in the draft:

- `producer_demand_trends.csv` and `producer_next_week_forecast.csv` support producer-facing forecast charts.
- They are descriptive proof-of-concept trend/forecast evidence from synthetic order history, not ARIMA/LSTM or production forecasting.

Research novelty wording used in the draft:

- The novelty is a risk-aware evaluation framework, not a new deep model.
- Cost-sensitive threshold optimisation formalises the automation versus food-safety trade-off.
- Quality-score ablation tests rule stability without claiming supervised grade accuracy.
- Producer-fair re-ranking analysis formalises recommendation exposure risk without changing the live default recommender.

Research figure exports:

- `docs/report_figures/research_novelty/cost_sensitive_threshold_tradeoff.png`
- `docs/report_figures/research_novelty/quality_score_ablation_grade_distribution.png`
- `docs/report_figures/research_novelty/producer_fair_reranking_tradeoff.png`
- `docs/report_figures/research_novelty/producer_next_week_forecast_chart.png`

The appendix is placed after the bibliography. It uses simple image filenames so the report works cleanly in Overleaf when the image files are uploaded into the same folder as `main.tex`:

- `efficientnetb0_aug_oversampled_finetuned_wsl_training_curves.png`
- `efficientnetb0_aug_oversampled_finetuned_wsl_confusion_matrix.png`
- `resnet50_aug_oversampled_finetuned_wsl_confusion_matrix.png`
- `00_Banana__Rotten.png`
- `00_true_Bellpepper__Rotten_pred_Bellpepper__Healthy.png`
- `00_Bellpepper__Rotten.png`
- `05_mid_banana_model_comparison.png`
- `cost_sensitive_threshold_tradeoff.png`
- `quality_score_ablation_grade_distribution.png`
- `producer_fair_reranking_tradeoff.png`
- `producer_next_week_forecast_chart.png`

Original repository locations:

- `outputs/figures/efficientnetb0_aug_oversampled_finetuned_wsl_training_curves.png`
- `outputs/confusion_matrices/efficientnetb0_aug_oversampled_finetuned_wsl_confusion_matrix.png`
- `outputs/confusion_matrices/resnet50_aug_oversampled_finetuned_wsl_confusion_matrix.png`
- `outputs/xai_examples/model_comparison/00_Banana__Rotten.png`
- `outputs/xai_examples/failure_analysis/00_true_Bellpepper__Rotten_pred_Bellpepper__Healthy.png`
- `outputs/xai_examples/xai_corrected_before_after/00_Bellpepper__Rotten.png`
- `outputs/xai_examples/custom_image_tests/05_mid_banana_model_comparison.png`
- `docs/report_figures/research_novelty/cost_sensitive_threshold_tradeoff.png`
- `docs/report_figures/research_novelty/quality_score_ablation_grade_distribution.png`
- `docs/report_figures/research_novelty/producer_fair_reranking_tradeoff.png`
- `docs/report_figures/research_novelty/producer_next_week_forecast_chart.png`
