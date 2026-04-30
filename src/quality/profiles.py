from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


PROFILE_PATH = Path(__file__).with_name("quality_profiles.yaml")


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    if text in {"", "{}"}:
        return {} if text == "{}" else ""
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def _parse_inline_range(value: str) -> list[float]:
    stripped = value.strip().strip("[]")
    return [float(part.strip()) for part in stripped.split(",") if part.strip()]


def _load_simple_yaml(path: Path) -> dict[str, dict[str, Any]]:
    """Parse the small profile YAML used by this project without adding PyYAML."""
    profiles: dict[str, dict[str, Any]] = {}
    current_profile: str | None = None
    current_colour: str | None = None
    in_hue_ranges = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].rstrip()
        if not line:
            continue

        indent = len(line) - len(line.lstrip(" "))
        text = line.strip()

        if indent == 0 and text.endswith(":"):
            current_profile = text[:-1].lower()
            profiles[current_profile] = {}
            current_colour = None
            in_hue_ranges = False
            continue

        if current_profile is None:
            continue

        if indent == 2 and text == "accepted_hue_ranges:":
            profiles[current_profile]["accepted_hue_ranges"] = {}
            in_hue_ranges = True
            current_colour = None
            continue

        if in_hue_ranges and indent == 4 and text.endswith(":"):
            current_colour = text[:-1]
            profiles[current_profile]["accepted_hue_ranges"][current_colour] = []
            continue

        if in_hue_ranges and indent == 6 and text.startswith("-"):
            if current_colour is not None:
                profiles[current_profile]["accepted_hue_ranges"][current_colour].append(
                    _parse_inline_range(text[1:])
                )
            continue

        if indent == 2 and ":" in text:
            key, value = text.split(":", maxsplit=1)
            profiles[current_profile][key.strip()] = _parse_scalar(value)
            in_hue_ranges = False
            current_colour = None

    return profiles


@lru_cache(maxsize=1)
def load_quality_profiles(path: Path = PROFILE_PATH) -> dict[str, dict[str, Any]]:
    return _load_simple_yaml(path)


def normalize_product_type(product_type: str) -> str:
    return product_type.lower().replace("__", "_").replace(" ", "_")


def get_quality_profile(product_type: str) -> tuple[dict[str, Any], list[str]]:
    profiles = load_quality_profiles()
    normalized = normalize_product_type(product_type)
    if normalized in profiles:
        return profiles[normalized], []

    return profiles["default"], ["unknown_produce_profile_used"]

