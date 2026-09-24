"""Shared helpers for PhotoMapper notebooks.

Author: NUS PhotoMapper Team
Date: 2026-09-24
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd


def normalize_text(value) -> str:
    """Normalize worksheet text for robust schema matching."""
    if pd.isna(value):
        text = ""
    else:
        text = str(value)
    text = text.lower().strip()
    text = text.replace("κ", "kappa")
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def detect_header_row(raw_df: pd.DataFrame) -> int:
    """Locate the worksheet row that contains indicator and model headers."""
    for idx in range(len(raw_df)):
        row_vals = [normalize_text(v) for v in raw_df.iloc[idx].tolist()]
        joined = " ".join(row_vals)
        has_indicator = any("indicatorname" in v or v == "indicatorname" for v in row_vals)
        has_qwen = "qwen36" in joined or any("qwen" in v for v in row_vals)
        has_intern = "internvl3" in joined or any("internvl" in v for v in row_vals)
        if has_indicator and has_qwen and has_intern:
            return idx
    raise ValueError("Could not detect header row.")


def detect_columns(header_values: Iterable) -> dict[str, object]:
    """Identify required indicator and model metric columns from a header row."""
    header_list = list(header_values)

    def find_col(contains_all):
        for col_name in header_list:
            normalized = normalize_text(col_name)
            if all(token in normalized for token in contains_all):
                return col_name
        return None

    no_col = find_col(["no"])
    indicator_col = find_col(["indicator", "name"])

    qwen_kappa = find_col(["qwen", "kappaw"]) or find_col(["qwen", "kw"])
    qwen_agreement = find_col(["qwen", "agr"]) or find_col(["qwen", "agreement"])
    intern_kappa = find_col(["internvl", "kappaw"]) or find_col(["internvl", "kw"])
    intern_agreement = find_col(["internvl", "agr"]) or find_col(["internvl", "agreement"])

    required = {
        "no_col": no_col,
        "indicator_col": indicator_col,
        "Qwen3.6-27B_kappa": qwen_kappa,
        "Qwen3.6-27B_agr": qwen_agreement,
        "InternVL3-8B_kappa": intern_kappa,
        "InternVL3-8B_agr": intern_agreement,
    }

    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Missing required detected columns: {missing}")

    return required


def geographic_cue_from_indicator(indicator_no: int):
    """Map indicator number to geographic cue category."""
    if 1 <= indicator_no <= 6:
        return "Building and structure"
    if 7 <= indicator_no <= 12:
        return "Road and transport"
    if 13 <= indicator_no <= 18:
        return "Vegetation and terrain"
    if 19 <= indicator_no <= 24:
        return "Objects and facilities"
    if 25 <= indicator_no <= 30:
        return "Global scene"
    return np.nan


def reasoning_evidence_from_indicator(indicator_no: int) -> str:
    """Map indicator number to reasoning evidence category."""
    position = (indicator_no - 1) % 6 + 1
    if position in (1, 2):
        return "Appearance"
    if position in (3, 4):
        return "Spatial layout"
    return "Landmark cue"
