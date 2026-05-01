from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_EVAL_DIR = PROJECT_ROOT / "outputs" / "final_evaluation"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _copy_if_exists(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return True


def _load_metadata() -> dict:
    metadata_path = PROJECT_ROOT / "models" / "model_metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _build_model_comparison_summary() -> pd.DataFrame:
    source = PROJECT_ROOT / "docs" / "report_figures" / "final_model_decision.csv"
    df = _read_csv(source)
    if df.empty:
        return df

    def suitability(row: pd.Series) -> str:
        if row["decision"] == "Selected final deployment candidate":
            return (
                "Selected: strong grouped metrics, fewer high-confidence errors, "
                "and lower serving risk."
            )
        if "challenger" in str(row["decision"]).lower():
            return (
                "Main challenger: stronger macro F1, but higher high-confidence "
                "error count increases deployment risk."
            )
        return "Comparison/ablation evidence; not selected for final demo."

    df = df.copy()
    df["deployment_suitability"] = df.apply(suitability, axis=1)
    return df


def _build_final_metrics(metadata: dict) -> dict:
    return {
        "final_model": metadata.get("model_name"),
        "selected_as_final": metadata.get("selected_as_final"),
        "base_model_family": metadata.get("base_model_family"),
        "split_strategy": metadata.get("split_strategy"),
        "split_dir": metadata.get("split_dir"),
        "metrics": metadata.get("metrics", {}),
        "risk_audit": metadata.get("risk_audit", {}),
        "quality_grading_layer": metadata.get("quality_grading_layer", {}),
        "inference_safeguards": metadata.get("inference_safeguards", {}),
        "selection_reason": metadata.get("selection_reason"),
        "selection_caveats": metadata.get("selection_caveats", {}),
        "artifacts": metadata.get("artifacts", {}),
    }


def _build_weak_class_summary() -> pd.DataFrame:
    report_paths = {
        "EfficientNetB0 final": PROJECT_ROOT
        / "outputs"
        / "figures"
        / "efficientnetb0_aug_oversampled_finetuned_wsl_classification_report.csv",
        "ResNet50 fine-tuned": PROJECT_ROOT
        / "outputs"
        / "figures"
        / "resnet50_aug_oversampled_finetuned_wsl_classification_report.csv",
    }
    rows = []
    for model, path in report_paths.items():
        report = _read_csv(path)
        if report.empty:
            continue
        class_column = report.columns[0]
        report = report.rename(columns={class_column: "class_name"})
        report = report[report["class_name"].astype(str).str.contains("__", regex=False)]
        weakest = report.sort_values("f1-score", ascending=True).head(5)
        for _, row in weakest.iterrows():
            rows.append(
                {
                    "model": model,
                    "class_name": row["class_name"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1_score": row["f1-score"],
                    "support": row["support"],
                }
            )
    return pd.DataFrame(rows)


def _build_quality_rule_summary() -> pd.DataFrame:
    grade_distribution = _read_csv(
        PROJECT_ROOT / "outputs" / "quality_rule_eval" / "grade_distribution.csv"
    )
    manual_review_rate = _read_csv(
        PROJECT_ROOT / "outputs" / "quality_rule_eval" / "manual_review_rate.csv"
    )
    action_distribution = _read_csv(
        PROJECT_ROOT / "outputs" / "quality_rule_eval" / "action_distribution.csv"
    )

    rows = []
    for model in sorted(
        set(grade_distribution.get("model", []))
        | set(manual_review_rate.get("model", []))
        | set(action_distribution.get("model", []))
    ):
        model_grades = grade_distribution[grade_distribution["model"] == model]
        model_actions = action_distribution[action_distribution["model"] == model]
        review_rows = manual_review_rate[manual_review_rate["model"] == model]
        rows.append(
            {
                "model": model,
                "grade_distribution": "; ".join(
                    f"{row.quality_grade}:{row.count}"
                    for row in model_grades.itertuples(index=False)
                ),
                "action_distribution": "; ".join(
                    f"{row.recommended_action}:{row.count}"
                    for row in model_actions.itertuples(index=False)
                ),
                "manual_review_rate": (
                    float(review_rows["manual_review_rate"].iloc[0])
                    if not review_rows.empty
                    else None
                ),
                "interpretation": (
                    "Rule-layer behaviour only; no supervised grade accuracy claim."
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_risky_case_examples(validation: pd.DataFrame) -> pd.DataFrame:
    if validation.empty:
        return validation

    df = validation.copy()
    if "confidence" in df.columns:
        low_confidence = df["confidence"] < 0.60
    else:
        low_confidence = False
    if "top1_top2_margin" in df.columns:
        low_margin = df["top1_top2_margin"] < 0.15
    else:
        low_margin = False
    wrong_prediction = (
        df["full_class_correct"].eq(False)
        if "full_class_correct" in df.columns
        else False
    )
    manual_review = (
        df["manual_review_required"].eq(True)
        if "manual_review_required" in df.columns
        else False
    )
    conservative_grade = (
        df["quality_grade"].isin(["C", "Review"])
        if "quality_grade" in df.columns
        else False
    )

    risky = df[wrong_prediction | low_confidence | low_margin | manual_review | conservative_grade]
    columns = [
        "image_id",
        "image_path",
        "expected_class",
        "model",
        "predicted_class",
        "confidence",
        "top1_top2_margin",
        "full_class_correct",
        "quality_grade",
        "manual_review_required",
        "recommended_action",
        "reason_codes",
        "warnings",
    ]
    return risky[[col for col in columns if col in risky.columns]]


def _write_adoption_recommendation(metadata: dict) -> None:
    content = f"""# Adoption Recommendation

## Recommendation

Adopt `{metadata.get("model_name", "the selected EfficientNetB0 model")}` as a
decision-support classifier for the BRFN demo, not as an autonomous quality
approval authority.

## Why This Is The Final Candidate

- Uses the grouped source-image split to reduce leakage from offline-augmented
  variants crossing train/validation/test.
- Achieves strong grouped-split performance: accuracy
  `{metadata.get("metrics", {}).get("test_accuracy")}`, macro F1
  `{metadata.get("metrics", {}).get("macro_f1")}`, weighted F1
  `{metadata.get("metrics", {}).get("weighted_f1")}`.
- Has fewer grouped XAI-audit high-confidence errors than the ResNet50
  fine-tuned challenger.
- Exposes confidence, top-1/top-2 margin, quality grade, action, reason codes,
  warnings, and manual-review flags for non-technical review.

## Deployment Position

Use the system to prioritise producer/admin inspection and marketplace handling:

- `A`: normal listing.
- `B`: discount or quick-sale handling.
- `C`: block pending manual review or discard decision.
- `Review`: do not make an automated listing decision.

## Evidence Caveat

The quality grade is produced by a transparent rule-based layer, not by a
supervised grade model. There are no expert Grade A/B/C labels in the dataset,
so the final evaluation reports grade distribution and risk-control behaviour,
not grade accuracy.

## Minimum Demo Story

1. Show `/health` and `/model-info`.
2. Predict a healthy image and explain model evidence plus quality layer output.
3. Predict a rotten or ambiguous image and show conservative handling.
4. Submit `/feedback` to demonstrate accountability and future monitoring.
"""
    (FINAL_EVAL_DIR / "adoption_recommendation.md").write_text(
        content,
        encoding="utf-8",
    )


def build_final_evaluation_pack() -> dict[str, str]:
    FINAL_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    metadata = _load_metadata()
    written: dict[str, str] = {}

    model_comparison = _build_model_comparison_summary()
    if not model_comparison.empty:
        path = FINAL_EVAL_DIR / "model_comparison_summary.csv"
        model_comparison.to_csv(path, index=False)
        written[path.name] = str(path)

    final_metrics = _build_final_metrics(metadata)
    path = FINAL_EVAL_DIR / "final_model_metrics.json"
    _write_json(path, final_metrics)
    written[path.name] = str(path)

    weak_classes = _build_weak_class_summary()
    if not weak_classes.empty:
        path = FINAL_EVAL_DIR / "weak_class_summary.csv"
        weak_classes.to_csv(path, index=False)
        written[path.name] = str(path)

    for source_name, destination_name in [
        ("grade_distribution.csv", "grade_distribution.csv"),
        ("manual_review_rate.csv", "manual_review_rate.csv"),
        ("weight_sensitivity.csv", "weight_sensitivity_summary.csv"),
        ("action_distribution.csv", "action_distribution.csv"),
    ]:
        copied = _copy_if_exists(
            PROJECT_ROOT / "outputs" / "quality_rule_eval" / source_name,
            FINAL_EVAL_DIR / destination_name,
        )
        if copied:
            written[destination_name] = str(FINAL_EVAL_DIR / destination_name)

    quality_summary = _build_quality_rule_summary()
    if not quality_summary.empty:
        path = FINAL_EVAL_DIR / "quality_rule_summary.csv"
        quality_summary.to_csv(path, index=False)
        written[path.name] = str(path)

    validation = _read_csv(
        PROJECT_ROOT / "docs" / "report_figures" / "custom_image_validation.csv"
    )
    if not validation.empty:
        path = FINAL_EVAL_DIR / "external_image_validation.csv"
        validation.to_csv(path, index=False)
        written[path.name] = str(path)

        risky = _build_risky_case_examples(validation)
        path = FINAL_EVAL_DIR / "risky_case_examples.csv"
        risky.to_csv(path, index=False)
        written[path.name] = str(path)

    _write_adoption_recommendation(metadata)
    written["adoption_recommendation.md"] = str(
        FINAL_EVAL_DIR / "adoption_recommendation.md"
    )

    manifest_path = FINAL_EVAL_DIR / "final_evaluation_manifest.json"
    _write_json(manifest_path, {"generated_files": written})
    written[manifest_path.name] = str(manifest_path)
    return written


if __name__ == "__main__":
    generated = build_final_evaluation_pack()
    for name, path in generated.items():
        print(f"{name}: {path}")
