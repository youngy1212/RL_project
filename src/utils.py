"""Shared utility functions for preprocessing, training, and evaluation."""

from __future__ import annotations

import json
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def ensure_dir(path: str | os.PathLike) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def set_global_seed(seed: int | None) -> None:
    """Set Python, NumPy, and torch seeds when torch is available."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def normalize_column_name(value: object) -> str:
    """Normalize a column name for forgiving Korean/English keyword matching."""
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = re.sub(r"\s+", "", text)
    return re.sub(r"[^0-9a-z가-힣]", "", text)


def _column_score(column_norm: str, candidate_norm: str) -> int:
    """Return a rough match score between a normalized column and candidate."""
    if not column_norm or not candidate_norm:
        return 0
    if column_norm == candidate_norm:
        return 1000 + len(candidate_norm)
    if candidate_norm in column_norm:
        return 500 + len(candidate_norm)
    if len(column_norm) >= 3 and column_norm in candidate_norm:
        return 100 + len(column_norm)
    return 0


def detect_column(
    columns: Sequence[object],
    candidates: Sequence[str],
    label: str,
    required: bool = True,
) -> str | None:
    """Detect one column from a list of candidate keywords.

    Matching ignores whitespace, punctuation, parentheses, underscores, and
    case. If a required column cannot be detected, a friendly ValueError lists
    the available columns and the expected candidates.
    """
    normalized_columns = [(str(col), normalize_column_name(col)) for col in columns]
    normalized_candidates = [normalize_column_name(candidate) for candidate in candidates]

    best_col: str | None = None
    best_score = 0
    for original, col_norm in normalized_columns:
        if original.startswith("Unnamed") and not col_norm:
            continue
        for cand_norm in normalized_candidates:
            score = _column_score(col_norm, cand_norm)
            if score > best_score:
                best_score = score
                best_col = original

    if best_col is None and required:
        available = "\n".join(f"  - {col}" for col in columns)
        expected = ", ".join(candidates)
        raise ValueError(
            f"{label} 컬럼을 자동 탐지하지 못했습니다.\n"
            f"필요 후보: {expected}\n"
            f"현재 데이터의 columns:\n{available}"
        )
    return best_col


def detect_matching_columns(columns: Sequence[object], candidates: Sequence[str]) -> list[str]:
    """Return all columns that match any candidate keyword."""
    matches: list[str] = []
    normalized_candidates = [normalize_column_name(candidate) for candidate in candidates]
    for col in columns:
        original = str(col)
        col_norm = normalize_column_name(original)
        if not col_norm:
            continue
        if any(_column_score(col_norm, cand_norm) >= 500 for cand_norm in normalized_candidates):
            matches.append(original)
    return matches


def normalize_station_id(value: object) -> str | None:
    """Normalize station IDs such as 101.0 to '101' while preserving ST- IDs."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = unicodedata.normalize("NFKC", str(value)).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None

    text = re.sub(r"\s+", "", text)
    numeric_text = text.replace(",", "")
    numeric_text = re.sub(r"\.0+$", "", numeric_text)
    if re.fullmatch(r"\d+", numeric_text):
        return str(int(numeric_text))

    try:
        value_float = float(numeric_text)
        if value_float.is_integer():
            return str(int(value_float))
    except ValueError:
        pass

    return text.upper()


def extract_station_id_from_text(value: object) -> str | None:
    """Extract a station-like ID from a free-text station name as a fallback."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = unicodedata.normalize("NFKC", str(value)).strip()
    if not text:
        return None

    st_match = re.search(r"(ST\s*[-_]?\s*\d+)", text, flags=re.IGNORECASE)
    if st_match:
        return normalize_station_id(st_match.group(1).replace(" ", "").replace("_", "-"))

    number_match = re.search(r"\b(\d{2,6})\b", text)
    if number_match:
        return normalize_station_id(number_match.group(1))
    return None


def station_id_digits(value: object) -> str | None:
    """Return the digit part of a station ID for fallback matching."""
    station_id = normalize_station_id(value)
    if not station_id:
        return None
    digits = re.sub(r"\D", "", station_id)
    return digits or None


def normalize_station_name(value: object) -> str | None:
    """Normalize station names for conservative fallback matching."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return None
    text = re.sub(r"^\d+\s*[.\-]?\s*", "", text)
    text = re.sub(r"[^0-9a-z가-힣]", "", text)
    return text or None


def numeric_series(series: pd.Series) -> pd.Series:
    """Convert a messy numeric series to float values."""
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace(r"[^\d.\-]", "", regex=True)
    )
    cleaned = cleaned.replace({"": np.nan, ".": np.nan, "-": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def haversine_distance_matrix(latitudes: Iterable[float], longitudes: Iterable[float]) -> np.ndarray:
    """Compute pairwise haversine distance in kilometers."""
    lat = np.radians(np.asarray(list(latitudes), dtype=float))
    lon = np.radians(np.asarray(list(longitudes), dtype=float))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return (6371.0088 * c).astype(np.float32)


def save_json(path: str | os.PathLike, payload: dict) -> None:
    """Save a dictionary as UTF-8 JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def load_time_features(csv_path: str | os.PathLike | None) -> np.ndarray | None:
    """Load a time_index CSV and return cyclic hour/day features."""
    if csv_path is None or not Path(csv_path).exists():
        return None
    frame = pd.read_csv(csv_path)
    if frame.empty:
        return None
    time_col = "time_bin" if "time_bin" in frame.columns else frame.columns[0]
    dt = pd.to_datetime(frame[time_col], errors="coerce")
    if dt.isna().all():
        return None

    hour = dt.dt.hour.to_numpy(dtype=float) + dt.dt.minute.to_numpy(dtype=float) / 60.0
    day = dt.dt.dayofweek.to_numpy(dtype=float)
    hour_angle = 2.0 * np.pi * hour / 24.0
    day_angle = 2.0 * np.pi * day / 7.0
    features = np.column_stack(
        [
            np.sin(hour_angle),
            np.cos(hour_angle),
            np.sin(day_angle),
            np.cos(day_angle),
        ]
    )
    return features.astype(np.float32)
