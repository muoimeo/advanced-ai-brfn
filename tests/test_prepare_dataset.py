import pandas as pd

from src.data.prepare_dataset import (
    audit_source_image_overlap,
    source_image_id_from_relative_path,
)


def test_source_image_id_normalizes_known_offline_augmentation_names():
    assert (
        source_image_id_from_relative_path(
            "Apple__Healthy/rotated_by_45_Screen Shot 2018-06-08.png"
        )
        == "Apple__Healthy/Screen Shot 2018-06-08.png"
    )
    assert (
        source_image_id_from_relative_path("Banana__Rotten/saltandpepper_sample.jpg")
        == "Banana__Rotten/sample.jpg"
    )


def test_audit_source_image_overlap_finds_cross_split_variants():
    train_df = pd.DataFrame(
        [
            {
                "relative_path": "Apple__Healthy/rotated_by_45_sample.png",
                "class_name": "Apple__Healthy",
            }
        ]
    )
    test_df = pd.DataFrame(
        [
            {
                "relative_path": "Apple__Healthy/sample.png",
                "class_name": "Apple__Healthy",
            }
        ]
    )

    overlap = audit_source_image_overlap({"train": train_df, "test": test_df})

    assert len(overlap) == 2
    assert overlap["source_image_id"].nunique() == 1
