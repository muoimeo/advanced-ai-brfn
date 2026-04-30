from io import BytesIO

import numpy as np
from PIL import Image

from src.quality.image_features import extract_image_features, rgb_to_hsv


def image_bytes(color: tuple[int, int, int], size=(64, 64)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_rgb_to_hsv_uses_degrees_and_unit_saturation_value():
    hsv = rgb_to_hsv(np.array([[[1.0, 0.0, 0.0]]]))

    assert hsv[0, 0, 0] == 0.0
    assert hsv[0, 0, 1] == 1.0
    assert hsv[0, 0, 2] == 1.0


def test_extract_image_features_returns_bounded_scores():
    features = extract_image_features(image_bytes((245, 220, 40)), "banana")

    assert 0.0 <= features.image_quality_score <= 100.0
    assert 0.0 <= features.color_score <= 100.0
    assert 0.0 <= features.size_proxy_score <= 100.0
    assert "colour_score_is_lighting_sensitive_proxy" in features.feature_warnings


def test_unknown_profile_uses_default_warning():
    features = extract_image_features(image_bytes((245, 220, 40)), "dragonfruit")

    assert "unknown_produce_profile_used" in features.feature_warnings


def test_dark_image_increases_dark_ratio_and_warns():
    features = extract_image_features(image_bytes((10, 8, 5)), "banana")

    assert features.dark_ratio >= 0.9
    assert "foreground_mask_low_confidence" in features.feature_warnings
