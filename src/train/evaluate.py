from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, test_ds, class_names: list[str]):
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.tolist())

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    return {
        "confusion_matrix": cm,
        "classification_report": report,
        "y_true": y_true,
        "y_pred": y_pred,
    }