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
- Capture one produce type per image.
- Place the produce close to the camera so it occupies most of the frame.
- Use a simple background where possible.
- Avoid strong reflections, glossy black surfaces, shadows, and clutter.
- Avoid extra produce items, plates with strong edges, labels, or packaging in
  the frame unless the goal is explicitly to test robustness.
- Use sufficient natural or diffuse lighting.
- Crop the image before validation if the produce item is small in the frame.
- Use varied lighting/backgrounds for stress testing, but label those examples
  as external robustness checks rather than normal operating conditions.

Why capture quality matters
---------------------------

The grouped test split measures performance on images from the same dataset
distribution. Household photos can introduce domain shift: different cameras,
lighting, backgrounds, reflections, object scale, and aspect ratio. Grad-CAM
and custom-image tests have shown that these conditions can make the model
attend to background regions or confuse produce types.

This is not treated as a reason to restart broad training. It is deployment
risk evidence. The current mitigation is:

- stricter manual-review threshold for low confidence;
- manual review when top-1 and top-2 predictions are too close;
- capture guidance for DESD users;
- feedback logging for future retraining evidence.

After adding images
-------------------

Run `notebooks/custom_image_test.ipynb` from VSCode WSL. Use:

```text
docs/report_figures/custom_image_validation.csv
docs/report_figures/custom_image_validation_summary.csv
outputs/xai_examples/custom_image_tests/
```

Treat these outputs as limited external-validation evidence, not as a full
deployment benchmark.
