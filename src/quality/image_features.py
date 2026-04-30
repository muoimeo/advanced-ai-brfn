from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from src.quality.profiles import get_quality_profile
from src.quality.schemas import ImageQualityFeatures


def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return float(max(lower, min(upper, value)))


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB values in 0..1 to HSV with hue in degrees and S/V in 0..1."""
    rgb = np.asarray(rgb, dtype=np.float32)
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]

    max_channel = np.max(rgb, axis=-1)
    min_channel = np.min(rgb, axis=-1)
    delta = max_channel - min_channel

    hue = np.zeros_like(max_channel)
    nonzero_delta = delta > 1e-8

    red_is_max = (max_channel == red) & nonzero_delta
    green_is_max = (max_channel == green) & nonzero_delta
    blue_is_max = (max_channel == blue) & nonzero_delta

    hue[red_is_max] = (
        60.0 * ((green[red_is_max] - blue[red_is_max]) / delta[red_is_max])
    ) % 360.0
    hue[green_is_max] = (
        60.0 * ((blue[green_is_max] - red[green_is_max]) / delta[green_is_max]) + 120.0
    )
    hue[blue_is_max] = (
        60.0 * ((red[blue_is_max] - green[blue_is_max]) / delta[blue_is_max]) + 240.0
    )

    saturation = np.zeros_like(max_channel)
    nonzero_value = max_channel > 1e-8
    saturation[nonzero_value] = delta[nonzero_value] / max_channel[nonzero_value]

    return np.stack([hue, saturation, max_channel], axis=-1)


def image_bytes_to_rgb_array(image_bytes: bytes, max_side: int = 512) -> np.ndarray:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((max_side, max_side))
    return np.asarray(image, dtype=np.float32) / 255.0


def _hue_in_ranges(hue: np.ndarray, accepted_hue_ranges: dict) -> np.ndarray:
    accepted = np.zeros_like(hue, dtype=bool)
    for ranges in accepted_hue_ranges.values():
        for start, end in ranges:
            if start <= end:
                accepted |= (hue >= start) & (hue <= end)
            else:
                accepted |= (hue >= start) | (hue <= end)
    return accepted


def _blur_score_from_rgb(rgb: np.ndarray) -> float:
    gray = (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    )
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0

    center = gray[1:-1, 1:-1]
    laplacian = (
        -4.0 * center
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return clamp(float(np.var(laplacian) * 1000.0))


def _brightness_score(value: np.ndarray) -> float:
    mean_value = float(np.mean(value))
    # Best around moderately bright images; penalise very dark or blown-out input.
    return clamp(100.0 - abs(mean_value - 0.55) * 180.0)


def _size_proxy_score(foreground_area_ratio: float) -> float:
    if 0.25 <= foreground_area_ratio <= 0.75:
        return 90.0
    if 0.15 <= foreground_area_ratio < 0.25:
        return 75.0
    if 0.75 < foreground_area_ratio <= 0.85:
        return 75.0
    return 60.0


def extract_image_features(
    image_bytes: bytes,
    product_type: str,
) -> ImageQualityFeatures:
    rgb = image_bytes_to_rgb_array(image_bytes)
    hsv = rgb_to_hsv(rgb)
    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]

    profile, warnings = get_quality_profile(product_type)
    min_saturation = float(profile.get("min_saturation", 0.18))
    min_value = float(profile.get("min_value", 0.12))
    max_value = float(profile.get("max_value", 0.97))
    dark_ratio_warn = float(profile.get("dark_ratio_warn", 0.18))
    dark_ratio_fail = float(profile.get("dark_ratio_fail", 0.32))

    foreground_mask = (
        (saturation >= min_saturation)
        & (value >= min_value)
        & (value <= max_value)
    )
    foreground_area_ratio = float(np.mean(foreground_mask))

    if np.any(foreground_mask):
        accepted_mask = _hue_in_ranges(
            hue,
            profile.get("accepted_hue_ranges", {}),
        ) & foreground_mask
        accepted_color_ratio = float(np.sum(accepted_mask) / np.sum(foreground_mask))
        dark_ratio = float(np.mean(value[foreground_mask] < dark_ratio_warn))
    else:
        accepted_color_ratio = 0.0
        dark_ratio = 1.0
        warnings.append("foreground_mask_low_confidence")

    color_score = accepted_color_ratio * 100.0
    if dark_ratio >= dark_ratio_fail:
        color_score -= 40.0
        warnings.append("dark_ratio_fail_threshold_reached")
    elif dark_ratio >= dark_ratio_warn:
        color_score -= 20.0
        warnings.append("dark_ratio_warn_threshold_reached")
    if "unknown_produce_profile_used" in warnings:
        color_score -= 10.0

    blur_score = _blur_score_from_rgb(rgb)
    brightness_score = _brightness_score(value)
    image_quality_score = clamp(0.60 * blur_score + 0.40 * brightness_score)
    size_proxy_score = _size_proxy_score(foreground_area_ratio)

    warnings.append("size_proxy_is_relative_to_image_area_not_physical_size")
    warnings.append("colour_score_is_lighting_sensitive_proxy")
    if foreground_area_ratio < 0.15:
        warnings.append("low_foreground_coverage")
    elif foreground_area_ratio > 0.85:
        warnings.append("high_foreground_coverage_or_background_mask_noise")

    return ImageQualityFeatures(
        image_quality_score=clamp(image_quality_score),
        color_score=clamp(color_score),
        defect_absence_score=clamp(100.0 - dark_ratio * 100.0),
        size_proxy_score=clamp(size_proxy_score),
        dark_ratio=dark_ratio,
        accepted_color_ratio=accepted_color_ratio,
        blur_score=clamp(blur_score),
        brightness_score=clamp(brightness_score),
        foreground_area_ratio=foreground_area_ratio,
        feature_warnings=list(dict.fromkeys(warnings)),
    )

