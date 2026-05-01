# Adoption Recommendation

## Recommendation

Adopt `efficientnetb0_aug_oversampled_finetuned_wsl` as a
decision-support classifier for the BRFN demo, not as an autonomous quality
approval authority.

## Why This Is The Final Candidate

- Uses the grouped source-image split to reduce leakage from offline-augmented
  variants crossing train/validation/test.
- Achieves strong grouped-split performance: accuracy
  `0.9719626168224299`, macro F1
  `0.9612978974472393`, weighted F1
  `0.9718550099623252`.
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
