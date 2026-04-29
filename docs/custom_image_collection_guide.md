Custom Image Collection Guide
=============================

Use this guide to strengthen external validation without overstating the result.

Location
--------

Place external validation images in:

```text
data/raw/custom_test_images/
```

Track labels in:

```text
docs/report_figures/custom_image_manifest.csv
```

Current priority images
-----------------------

Collect or add clearly licensed external images for:

- healthy apple
- rotten apple
- healthy tomato
- rotten tomato
- healthy potato
- rotten potato
- healthy bell pepper
- rotten bell pepper

Why these classes
-----------------

The current custom set is banana-heavy. Apple, tomato, potato, and bell pepper
are better final-demo choices because they connect to weak-class and XAI failure
analysis evidence.

Rules
-----

- Prefer real user-collected photos or clearly licensed external photos.
- Do not use images copied from the training/test dataset.
- Do not use AI-generated images as robustness evidence. If generated images
  are ever used, label them as synthetic demo images only.
- Keep filenames stable and update `expected_class` in the manifest.
- Use varied lighting/backgrounds, but keep the produce item visible.

After adding images
-------------------

Run `notebooks/custom_image_test.ipynb` from VSCode WSL. Use the output CSV and
comparison figures as limited external-validation evidence, not as a full
deployment benchmark.
