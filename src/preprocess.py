"""Preprocess Seoul public bike OD and station files for RL training."""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config import (
    CSV_ENCODINGS,
    DEFAULT_FREQ,
    DEFAULT_OD_GLOB,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_STATION_GLOB,
    DEFAULT_TOP_N,
    DEFAULT_TRAIN_RATIO,
    OD_COUNT_CANDIDATES,
    OD_DATE_CANDIDATES,
    OD_END_ID_CANDIDATES,
    OD_END_NAME_CANDIDATES,
    OD_START_ID_CANDIDATES,
    OD_START_NAME_CANDIDATES,
    OD_TIME_CANDIDATES,
    STATION_CAPACITY_CANDIDATES,
    STATION_ID_CANDIDATES,
    STATION_LAT_CANDIDATES,
    STATION_LON_CANDIDATES,
    STATION_NAME_CANDIDATES,
)
from utils import (
    detect_column,
    detect_matching_columns,
    ensure_dir,
    extract_station_id_from_text,
    haversine_distance_matrix,
    normalize_station_name,
    normalize_station_id,
    numeric_series,
    save_json,
    station_id_digits,
)


SEOUL_DISTRICTS = [
    "종로구",
    "중구",
    "용산구",
    "성동구",
    "광진구",
    "동대문구",
    "중랑구",
    "성북구",
    "강북구",
    "도봉구",
    "노원구",
    "은평구",
    "서대문구",
    "마포구",
    "양천구",
    "강서구",
    "구로구",
    "금천구",
    "영등포구",
    "동작구",
    "관악구",
    "서초구",
    "강남구",
    "송파구",
    "강동구",
]

DISTRICT_PATTERN = re.compile("|".join(map(re.escape, SEOUL_DISTRICTS)))


def read_csv_sample(path: str | Path, nrows: int = 0) -> tuple[pd.DataFrame, str]:
    """Read a CSV sample by trying supported Korean encodings."""
    errors: list[str] = []
    for encoding in CSV_ENCODINGS:
        try:
            frame = pd.read_csv(path, encoding=encoding, nrows=nrows, low_memory=False)
            return frame, encoding
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError(f"CSV 인코딩을 판별하지 못했습니다: {path}\n" + "\n".join(errors))


def normalize_freq(freq: str) -> str:
    """Normalize legacy pandas offset aliases to current spellings."""
    freq = str(freq).strip()
    if freq == "H":
        return "1h"
    if freq.endswith("H") and freq[:-1].isdigit():
        return freq[:-1] + "h"
    if freq == "T":
        return "1min"
    if freq.endswith("T") and freq[:-1].isdigit():
        return freq[:-1] + "min"
    return freq


def detect_od_columns(columns: Iterable[object]) -> dict[str, str | None]:
    """Detect OD CSV columns needed for rental/return demand aggregation."""
    columns = list(columns)
    date_col = detect_column(columns, OD_DATE_CANDIDATES, "OD 날짜", required=False)
    time_or_datetime_col = detect_column(columns, OD_TIME_CANDIDATES, "OD 시간/일시", required=False)
    if not date_col and not time_or_datetime_col:
        available = "\n".join(f"  - {col}" for col in columns)
        raise ValueError(
            "OD 날짜/시간 컬럼을 자동 탐지하지 못했습니다. date_col 또는 datetime_col이 필요합니다.\n"
            f"현재 데이터의 columns:\n{available}"
        )
    if date_col and not time_or_datetime_col:
        available = "\n".join(f"  - {col}" for col in columns)
        raise ValueError(
            "OD 날짜 컬럼이 분리되어 있는 경우 time_col도 필요합니다.\n"
            f"현재 데이터의 columns:\n{available}"
        )

    datetime_col = None if date_col else time_or_datetime_col
    time_col = time_or_datetime_col if date_col and time_or_datetime_col != date_col else None
    detected = {
        "date_col": date_col,
        "datetime_col": datetime_col,
        "time_col": time_col,
        "start_station_id_col": detect_column(columns, OD_START_ID_CANDIDATES, "시작 대여소 ID"),
        "end_station_id_col": detect_column(columns, OD_END_ID_CANDIDATES, "종료 대여소 ID"),
        "start_station_name_col": detect_column(columns, OD_START_NAME_CANDIDATES, "시작 대여소명", required=False),
        "end_station_name_col": detect_column(columns, OD_END_NAME_CANDIDATES, "종료 대여소명", required=False),
        "count_col": detect_column(columns, OD_COUNT_CANDIDATES, "수요 건수", required=False),
    }
    return detected


def detect_station_columns(columns: Iterable[object]) -> dict[str, object]:
    """Detect station metadata columns from an Excel sheet."""
    columns = list(columns)
    station_id_col = detect_column(columns, STATION_ID_CANDIDATES, "대여소 ID/관리번호", required=False)
    station_name_col = detect_column(columns, STATION_NAME_CANDIDATES, "대여소명")
    detected = {
        "station_id_col": station_id_col,
        "station_name_col": station_name_col,
        "lat_col": detect_column(columns, STATION_LAT_CANDIDATES, "위도"),
        "lon_col": detect_column(columns, STATION_LON_CANDIDATES, "경도"),
        "capacity_cols": detect_matching_columns(columns, STATION_CAPACITY_CANDIDATES),
    }
    return detected


def _normalize_time_values(series: pd.Series) -> pd.Series:
    """Normalize HH, HMM, HHMM, or HHMMSS-like values to HHMM strings."""
    raw = series.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
    digits = raw.str.replace(r"\D", "", regex=True)

    def convert(value: str) -> str:
        if not value:
            return ""
        if len(value) <= 2:
            return value.zfill(2) + "00"
        if len(value) == 3:
            return "0" + value
        if len(value) == 4:
            return value
        if len(value) >= 6:
            return value[:4]
        return value.zfill(4)

    return digits.map(convert)


def _parse_direct_datetime(series: pd.Series) -> pd.Series:
    """Parse a datetime series with generic and digit-only fallbacks."""
    raw = series.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
    parsed = pd.to_datetime(raw, errors="coerce")
    digits = raw.str.replace(r"\D", "", regex=True)

    for length, fmt in ((14, "%Y%m%d%H%M%S"), (12, "%Y%m%d%H%M"), (8, "%Y%m%d")):
        mask = parsed.isna() & digits.str.len().eq(length)
        if mask.any():
            parsed.loc[mask] = pd.to_datetime(digits.loc[mask], format=fmt, errors="coerce")
    return parsed


def build_datetime_series(frame: pd.DataFrame, columns: dict[str, str | None]) -> pd.Series:
    """Build a datetime series from detected OD date/time columns."""
    date_col = columns.get("date_col")
    time_col = columns.get("time_col")
    datetime_col = columns.get("datetime_col")

    if datetime_col:
        parsed = _parse_direct_datetime(frame[datetime_col])
        if parsed.notna().any():
            return parsed

    if date_col and time_col and date_col != time_col:
        date_digits = (
            frame[date_col]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0+$", "", regex=True)
            .str.replace(r"\D", "", regex=True)
            .str[:8]
        )
        time_digits = _normalize_time_values(frame[time_col])
        combined = date_digits + time_digits
        parsed = pd.to_datetime(combined, format="%Y%m%d%H%M", errors="coerce")
        if parsed.notna().any():
            return parsed

    if time_col:
        parsed = _parse_direct_datetime(frame[time_col])
        if parsed.notna().any():
            return parsed

    if date_col:
        parsed = _parse_direct_datetime(frame[date_col])
        if parsed.notna().any():
            return parsed

    available = "\n".join(f"  - {col}" for col in frame.columns)
    raise ValueError(
        "OD 시간 정보를 datetime으로 변환하지 못했습니다. 날짜/시간 컬럼 형식을 확인해주세요.\n"
        f"탐지된 컬럼: {columns}\n현재 columns:\n{available}"
    )


def _required_od_columns(detected: dict[str, str | None]) -> list[str]:
    """Return unique non-null OD columns required for OD aggregation."""
    required = [
        detected.get("date_col"),
        detected.get("datetime_col"),
        detected.get("time_col"),
        detected.get("start_station_id_col"),
        detected.get("end_station_id_col"),
        detected.get("count_col"),
    ]
    return list(dict.fromkeys([col for col in required if col]))


def _optional_od_columns(detected: dict[str, str | None], actual_columns: Iterable[object]) -> list[str]:
    """Return optional OD columns that are present in the current file."""
    actual_column_set = {str(col) for col in actual_columns}
    optional_cols = []
    for col in [detected.get("start_station_name_col"), detected.get("end_station_name_col")]:
        if col and col in actual_column_set:
            optional_cols.append(col)
    return list(dict.fromkeys(optional_cols))


def _od_usecols_for_file(path: str, detected: dict[str, str | None], actual_columns: Iterable[object]) -> list[str]:
    """Validate per-file OD columns and return safe usecols."""
    actual_columns = [str(col) for col in actual_columns]
    required_cols = _required_od_columns(detected)
    missing_required = [col for col in required_cols if col not in actual_columns]
    if missing_required:
        raise ValueError(
            f"필수 컬럼이 파일 header에 없습니다. file={path}, "
            f"missing={missing_required}, columns={actual_columns}"
        )

    optional_cols = _optional_od_columns(detected, actual_columns)
    missing_optional = [
        col
        for col in [detected.get("start_station_name_col"), detected.get("end_station_name_col")]
        if col and col not in actual_columns
    ]
    for col in missing_optional:
        print(f"[WARN] 선택 컬럼이 없어 OD station_name 보조 매핑에서 제외합니다. file={path}, column={col}")
    return list(dict.fromkeys(required_cols + optional_cols))


def aggregate_od_files(
    od_paths: list[str],
    detected: dict[str, str | None],
    freq: str,
    chunk_size: int = 200_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int | float]]:
    """Aggregate OD files into rental and return demand by time bin/station."""
    rental_parts: list[pd.DataFrame] = []
    return_parts: list[pd.DataFrame] = []
    flow_parts: list[pd.DataFrame] = []
    name_parts: list[pd.DataFrame] = []
    total_rows = 0
    removed_rows = 0

    for path in od_paths:
        file_rows = 0
        header_frame, file_encoding = read_csv_sample(path, nrows=0)
        file_detected = detect_od_columns(header_frame.columns)
        usecols = _od_usecols_for_file(path, file_detected, header_frame.columns)
        file_has_start_name = bool(file_detected.get("start_station_name_col") in usecols)
        file_has_end_name = bool(file_detected.get("end_station_name_col") in usecols)
        if file_detected != detected:
            print(f"[Detect] OD columns for {Path(path).name}: {file_detected}")
        if not file_has_start_name:
            print("[WARN] 시작 대여소명 컬럼이 없어 station_name 매핑을 건너뜁니다.")
        if not file_has_end_name:
            print("[WARN] 종료 대여소명 컬럼이 없어 station_name 매핑을 건너뜁니다.")
        print(f"[OD] 읽는 중: {path} (encoding={file_encoding})")
        reader = pd.read_csv(
            path,
            encoding=file_encoding,
            usecols=usecols,
            dtype=str,
            chunksize=chunk_size,
            low_memory=False,
        )
        for chunk in reader:
            file_rows += len(chunk)
            total_rows += len(chunk)
            required = [
                file_detected["start_station_id_col"],
                file_detected["end_station_id_col"],
            ]
            if file_detected.get("date_col"):
                required.append(file_detected["date_col"])
            if file_detected.get("time_col"):
                required.append(file_detected["time_col"])
            if file_detected.get("datetime_col"):
                required.append(file_detected["datetime_col"])
            if file_detected.get("count_col"):
                required.append(file_detected["count_col"])

            missing_required = [col for col in required if col and col not in chunk.columns]
            if missing_required:
                raise ValueError(
                    f"필수 컬럼이 chunk에 없습니다. file={path}, "
                    f"missing={missing_required}, columns={list(chunk.columns)}"
                )

            start_id_col = file_detected["start_station_id_col"]
            end_id_col = file_detected["end_station_id_col"]
            count_col = file_detected.get("count_col")
            start_name_col = file_detected.get("start_station_name_col")
            end_name_col = file_detected.get("end_station_name_col")

            if start_name_col and start_name_col in chunk.columns:
                start_names = chunk[start_name_col]
            else:
                start_names = None

            if end_name_col and end_name_col in chunk.columns:
                end_names = chunk[end_name_col]
            else:
                end_names = None

            datetime_values = build_datetime_series(chunk, file_detected)
            start_ids = chunk[start_id_col].map(normalize_station_id)
            end_ids = chunk[end_id_col].map(normalize_station_id)

            if count_col:
                counts = numeric_series(chunk[count_col]).fillna(0.0)
            else:
                counts = pd.Series(1.0, index=chunk.index)

            work = pd.DataFrame(
                {
                    "time_bin": datetime_values.dt.floor(freq),
                    "start_station_id": start_ids,
                    "end_station_id": end_ids,
                    "count": counts.astype(float),
                }
            )
            valid = (
                work["time_bin"].notna()
                & work["start_station_id"].notna()
                & work["end_station_id"].notna()
                & (work["count"] > 0)
            )
            removed_rows += int((~valid).sum())
            work = work.loc[valid]
            if work.empty:
                continue

            if start_names is not None:
                names = pd.DataFrame(
                    {
                        "station_id": start_ids,
                        "station_name": start_names.astype(str),
                    }
                )
                name_parts.append(names.dropna().drop_duplicates())

            if end_names is not None:
                names = pd.DataFrame(
                    {
                        "station_id": end_ids,
                        "station_name": end_names.astype(str),
                    }
                )
                name_parts.append(names.dropna().drop_duplicates())

            flow_parts.append(
                work.groupby(["time_bin", "start_station_id", "end_station_id"], as_index=False)["count"].sum()
            )
            rental_parts.append(
                work.groupby(["time_bin", "start_station_id"], as_index=False)["count"]
                .sum()
                .rename(columns={"start_station_id": "station_id", "count": "rental_demand"})
            )
            return_parts.append(
                work.groupby(["time_bin", "end_station_id"], as_index=False)["count"]
                .sum()
                .rename(columns={"end_station_id": "station_id", "count": "return_demand"})
            )

        print(f"[OD] row 수: {Path(path).name} = {file_rows:,}")

    if not rental_parts or not return_parts:
        raise RuntimeError(
            "OD 전처리 결과가 비어 있습니다. 원인 후보: 시간 컬럼 파싱 실패, 대여소 ID 컬럼 불일치, "
            "건수 컬럼이 모두 0/결측, 또는 glob 패턴이 잘못됨."
        )

    rental = pd.concat(rental_parts, ignore_index=True)
    returns = pd.concat(return_parts, ignore_index=True)
    od_flows = pd.concat(flow_parts, ignore_index=True)
    rental = rental.groupby(["time_bin", "station_id"], as_index=False)["rental_demand"].sum()
    returns = returns.groupby(["time_bin", "station_id"], as_index=False)["return_demand"].sum()
    od_flows = od_flows.groupby(["time_bin", "start_station_id", "end_station_id"], as_index=False)["count"].sum()
    if name_parts:
        od_station_names = pd.concat(name_parts, ignore_index=True)
        od_station_names["station_name_norm"] = od_station_names["station_name"].map(normalize_station_name)
        od_station_names = od_station_names.loc[od_station_names["station_name_norm"].notna()]
        od_station_names = (
            od_station_names.groupby(["station_id", "station_name"], as_index=False)
            .size()
            .sort_values(["station_id", "size"], ascending=[True, False])
            .drop_duplicates("station_id", keep="first")
            .drop(columns=["size"])
            .reset_index(drop=True)
        )
    else:
        od_station_names = pd.DataFrame(columns=["station_id", "station_name"])

    stats = {
        "total_rows": int(total_rows),
        "removed_rows": int(removed_rows),
        "total_rental_demand": float(rental["rental_demand"].sum()),
        "total_return_demand": float(returns["return_demand"].sum()),
    }
    return rental, returns, od_flows, od_station_names, stats


def _flatten_excel_columns(columns: pd.Index | pd.MultiIndex) -> list[str]:
    """Flatten regular or multi-row Excel headers into unique string columns."""
    flattened: list[str] = []
    for col in columns:
        if isinstance(col, tuple):
            parts = []
            for part in col:
                text = "" if pd.isna(part) else str(part).strip()
                if not text or text.lower().startswith("unnamed"):
                    continue
                parts.append(text.replace("\n", " "))
            name = " ".join(parts).strip()
        else:
            name = str(col).strip().replace("\n", " ")
        flattened.append(name or "Unnamed")

    counts: dict[str, int] = {}
    unique: list[str] = []
    for name in flattened:
        counts[name] = counts.get(name, 0) + 1
        if counts[name] == 1:
            unique.append(name)
        else:
            unique.append(f"{name}_{counts[name]}")
    return unique


def _station_header_candidates() -> list[int | list[int]]:
    """Return single-row and multi-row header candidates for messy Excel files."""
    candidates: list[int | list[int]] = list(range(10))
    candidates.extend(
        [
            [0, 1],
            [0, 1, 2],
            [0, 1, 2, 3],
            [0, 1, 2, 3, 4],
            [1, 2, 3],
            [1, 2, 3, 4],
        ]
    )
    return candidates


def read_station_excel(station_paths: list[str]) -> tuple[pd.DataFrame, dict[str, object], str, int | list[int]]:
    """Read station Excel data and detect its header row and key columns."""
    last_error: Exception | None = None
    for path in station_paths:
        for header in _station_header_candidates():
            try:
                frame = pd.read_excel(path, header=header)
                frame.columns = _flatten_excel_columns(frame.columns)
                frame = frame.dropna(how="all").dropna(axis=1, how="all")
                if frame.empty:
                    continue
                detected = detect_station_columns(frame.columns)
                print(f"[Station] 사용 파일: {path}")
                print(f"[Station] 탐지 header row: {header}")
                return frame, detected, path, header
            except Exception as exc:
                last_error = exc
                continue
    raise ValueError(
        "대여소 정보 엑셀에서 필요한 컬럼을 찾지 못했습니다. "
        "대여소명, 위도, 경도, 대여소ID/관리번호 컬럼명을 확인해주세요.\n"
        f"마지막 오류: {last_error}"
    )


def build_station_meta(frame: pd.DataFrame, detected: dict[str, object]) -> pd.DataFrame:
    """Normalize station metadata and derive capacity."""
    id_col = detected.get("station_id_col")
    name_col = str(detected["station_name_col"])
    lat_col = str(detected["lat_col"])
    lon_col = str(detected["lon_col"])
    capacity_cols = list(detected.get("capacity_cols") or [])

    if id_col:
        station_ids = frame[str(id_col)].map(normalize_station_id)
    else:
        station_ids = pd.Series([None] * len(frame), index=frame.index)
    fallback_ids = frame[name_col].map(extract_station_id_from_text)
    station_ids = station_ids.where(station_ids.notna(), fallback_ids)

    if capacity_cols:
        capacity_values = pd.concat([numeric_series(frame[col]) for col in capacity_cols], axis=1)
        capacity = capacity_values.sum(axis=1, min_count=1).fillna(20.0)
    else:
        capacity = pd.Series(20.0, index=frame.index)
    capacity = capacity.where(capacity > 0, 20.0).fillna(20.0)

    meta = pd.DataFrame(
        {
            "station_id": station_ids,
            "station_name": frame[name_col].astype(str).str.strip(),
            "latitude": numeric_series(frame[lat_col]),
            "longitude": numeric_series(frame[lon_col]),
            "capacity": capacity.astype(float),
        }
    )
    before = len(meta)
    meta = meta.loc[meta["station_id"].notna()].copy()
    meta = meta.drop_duplicates("station_id", keep="first").reset_index(drop=True)
    print(f"[Station] station_id 없음으로 제거된 row 수: {before - len(meta):,}")
    return meta


def extract_district(value: object) -> str | None:
    """Extract a Seoul district name from a scalar value."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    match = DISTRICT_PATTERN.search(str(value))
    return match.group(0) if match else None


def district_extraction_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Count extractable district names for each string-like station column."""
    counts: dict[str, int] = {}
    for col in frame.columns:
        series = frame[col]
        if series.dtype == object or pd.api.types.is_string_dtype(series):
            counts[str(col)] = int(series.map(extract_district).notna().sum())
    return counts


def detect_district_series(frame: pd.DataFrame, district_col: str | None = None) -> tuple[pd.Series, str]:
    """Find station districts from a user column or all string-like columns."""
    if district_col:
        if district_col not in frame.columns:
            raise ValueError(f"--district-col 컬럼을 찾지 못했습니다: {district_col}, columns={list(frame.columns)}")
        return frame[district_col].map(extract_district), district_col

    counts = district_extraction_counts(frame)
    best_col = max(counts, key=counts.get, default=None)
    if best_col and counts[best_col] > 0:
        return frame[best_col].map(extract_district), best_col

    raise ValueError(
        "대여소 정보 엑셀에서 자치구명을 찾지 못했습니다. 주소/자치구 컬럼이 있는지 확인하거나 --district-col을 지정하세요."
    )


def attach_station_district(
    station_meta: pd.DataFrame,
    station_frame: pd.DataFrame,
    station_columns: dict[str, object],
    district_col: str | None = None,
) -> tuple[pd.DataFrame, str, int]:
    """Attach district_name to station metadata and drop stations without districts."""
    id_col = station_columns.get("station_id_col")
    name_col = str(station_columns["station_name_col"])
    district_series, source = detect_district_series(station_frame, district_col)

    if id_col:
        raw_ids = station_frame[str(id_col)].map(normalize_station_id)
    else:
        raw_ids = pd.Series([None] * len(station_frame), index=station_frame.index)
    fallback_ids = station_frame[name_col].map(extract_station_id_from_text)
    raw_ids = raw_ids.where(raw_ids.notna(), fallback_ids)

    district_map = (
        pd.DataFrame({"station_source_id": raw_ids, "district_name": district_series})
        .dropna(subset=["station_source_id"])
        .drop_duplicates("station_source_id", keep="first")
    )
    before = len(station_meta)
    merged = station_meta.merge(district_map, on="station_source_id", how="left")
    missing = int(merged["district_name"].isna().sum())
    merged = merged.loc[merged["district_name"].notna()].reset_index(drop=True)
    print(f"[District] 추출 source: {source}")
    print(f"[District] 자치구 미확인으로 제외된 station 수: {missing:,} / {before:,}")
    return merged, source, missing


def build_station_match_map(
    station_meta: pd.DataFrame,
    demand_station_ids: Iterable[str],
    od_station_names: pd.DataFrame | None = None,
) -> dict[str, pd.Series]:
    """Map OD station IDs to station metadata by ID, digits, or unique station name."""
    by_exact = {row["station_id"]: row for _, row in station_meta.iterrows()}

    digit_rows: dict[str, list[pd.Series]] = {}
    for _, row in station_meta.iterrows():
        digits = station_id_digits(row["station_id"])
        if digits:
            digit_rows.setdefault(digits, []).append(row)
    by_unique_digits = {digits: rows[0] for digits, rows in digit_rows.items() if len(rows) == 1}

    name_rows: dict[str, list[pd.Series]] = {}
    for _, row in station_meta.iterrows():
        name_norm = normalize_station_name(row["station_name"])
        if name_norm:
            name_rows.setdefault(name_norm, []).append(row)
    by_unique_name = {name: rows[0] for name, rows in name_rows.items() if len(rows) == 1}

    od_name_lookup: dict[str, str] = {}
    if od_station_names is not None and not od_station_names.empty:
        for _, row in od_station_names.iterrows():
            name_norm = normalize_station_name(row["station_name"])
            station_id = normalize_station_id(row["station_id"])
            if station_id and name_norm:
                od_name_lookup[station_id] = name_norm

    matches: dict[str, pd.Series] = {}
    for station_id in demand_station_ids:
        if station_id in by_exact:
            matches[station_id] = by_exact[station_id]
            continue
        digits = station_id_digits(station_id)
        if digits and digits in by_unique_digits:
            matches[station_id] = by_unique_digits[digits]
            continue
        name_norm = od_name_lookup.get(station_id)
        if name_norm and name_norm in by_unique_name:
            matches[station_id] = by_unique_name[name_norm]
    return matches


def select_stations(
    rental: pd.DataFrame,
    returns: pd.DataFrame,
    station_meta: pd.DataFrame,
    top_n: int,
    od_station_names: pd.DataFrame | None = None,
    use_all_stations: bool = False,
    min_station_demand: float = 1.0,
) -> tuple[list[str], pd.DataFrame, dict[str, float]]:
    """Select top-demand stations that can be matched to station metadata."""
    rental_totals = rental.groupby("station_id")["rental_demand"].sum()
    return_totals = returns.groupby("station_id")["return_demand"].sum()
    demand_totals = rental_totals.add(return_totals, fill_value=0.0).sort_values(ascending=False)
    demand_station_ids = list(demand_totals.index)

    matches = build_station_match_map(station_meta, demand_station_ids, od_station_names)
    matching_rate = len(matches) / max(1, len(demand_station_ids))
    print(f"[Match] OD demand station 수: {len(demand_station_ids):,}")
    print(f"[Match] station 정보와 매칭된 수: {len(matches):,} ({matching_rate:.2%})")

    candidate_rows = []
    for od_station_id, source_row in matches.items():
        if pd.isna(source_row["latitude"]) or pd.isna(source_row["longitude"]):
            continue
        capacity = float(source_row["capacity"]) if not pd.isna(source_row["capacity"]) else 20.0
        if capacity <= 0:
            continue
        candidate_rows.append(
            {
                "station_id": od_station_id,
                "station_source_id": source_row["station_id"],
                "station_name": source_row["station_name"],
                "latitude": float(source_row["latitude"]),
                "longitude": float(source_row["longitude"]),
                "capacity": capacity,
                "total_demand": float(demand_totals.get(od_station_id, 0.0)),
                "total_rental_demand": float(rental_totals.get(od_station_id, 0.0)),
                "total_return_demand": float(return_totals.get(od_station_id, 0.0)),
            }
        )

    selected_meta = pd.DataFrame(candidate_rows)
    if selected_meta.empty:
        raise RuntimeError(
            "선택 가능한 대여소가 없습니다. 원인 후보: OD station_id와 엑셀 station_id가 매칭되지 않음, "
            "위도/경도 결측, capacity 결측 처리 후 비정상 값."
        )

    selected_meta = selected_meta.loc[selected_meta["total_demand"] >= float(min_station_demand)].copy()
    if selected_meta.empty:
        raise RuntimeError(f"min_station_demand={min_station_demand} 이상인 매칭 대여소가 없습니다.")

    selected_meta = selected_meta.sort_values("total_demand", ascending=False).reset_index(drop=True)
    if use_all_stations:
        print(f"[Select] use_all_stations=True, 매칭 가능한 전체 {len(selected_meta):,}개 대여소를 사용합니다.")
    else:
        selected_meta = selected_meta.head(top_n).reset_index(drop=True)
        if len(selected_meta) < top_n:
            print(f"[Warning] top_n={top_n}보다 적은 {len(selected_meta)}개 대여소만 사용합니다.")

    selected_ids = selected_meta["station_id"].tolist()
    stats = {
        "od_unique_station_count": float(len(demand_station_ids)),
        "matched_station_count": float(len(matches)),
        "station_matching_rate": float(matching_rate),
    }
    if use_all_stations:
        print(f"[Select] 선택된 station 수: {len(selected_ids):,}")
    else:
        print("[Select] 선택된 station_id:", ", ".join(selected_ids))
    return selected_ids, selected_meta, stats


def build_demand_matrices(
    rental: pd.DataFrame,
    returns: pd.DataFrame,
    selected_ids: list[str],
    freq: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build continuous time-indexed rental and return matrices."""
    all_times = pd.Index(rental["time_bin"]).union(pd.Index(returns["time_bin"])).sort_values()
    if len(all_times) == 0:
        raise RuntimeError("time_bin이 비어 있습니다. 시간 컬럼 파싱 결과를 확인해주세요.")

    time_index = pd.date_range(start=all_times.min(), end=all_times.max(), freq=freq)
    rental_matrix = (
        rental.loc[rental["station_id"].isin(selected_ids)]
        .pivot(index="time_bin", columns="station_id", values="rental_demand")
        .reindex(index=time_index, columns=selected_ids)
        .fillna(0.0)
    )
    return_matrix = (
        returns.loc[returns["station_id"].isin(selected_ids)]
        .pivot(index="time_bin", columns="station_id", values="return_demand")
        .reindex(index=time_index, columns=selected_ids)
        .fillna(0.0)
    )
    if rental_matrix.empty or return_matrix.empty:
        raise RuntimeError("선택된 대여소의 demand matrix가 비어 있습니다. station_id 매칭 결과를 확인해주세요.")
    return rental_matrix, return_matrix


def summarize_od_district_flows(
    od_flows: pd.DataFrame,
    selected_meta: pd.DataFrame,
    drop_cross_district_od: bool,
) -> tuple[pd.DataFrame | None, dict[str, int | float]]:
    """Count same/cross-district OD flows and optionally keep only same-district flows."""
    if "district_name" not in selected_meta.columns:
        return None, {
            "same_district_od_count": 0,
            "cross_district_od_count": 0,
            "cross_district_od_ratio": 0.0,
        }

    district_lookup = selected_meta.set_index("station_id")["district_name"].to_dict()
    flows = od_flows.copy()
    flows["start_district"] = flows["start_station_id"].map(district_lookup)
    flows["end_district"] = flows["end_station_id"].map(district_lookup)
    flows = flows.loc[flows["start_district"].notna() & flows["end_district"].notna()].copy()
    if flows.empty:
        return flows, {
            "same_district_od_count": 0,
            "cross_district_od_count": 0,
            "cross_district_od_ratio": 0.0,
        }

    same_mask = flows["start_district"].eq(flows["end_district"])
    same_count = float(flows.loc[same_mask, "count"].sum())
    cross_count = float(flows.loc[~same_mask, "count"].sum())
    total = same_count + cross_count
    stats = {
        "same_district_od_count": int(same_count),
        "cross_district_od_count": int(cross_count),
        "cross_district_od_ratio": float(cross_count / max(1.0, total)),
    }
    if drop_cross_district_od:
        flows = flows.loc[same_mask].copy()
    return flows, stats


def demand_from_flows(od_flows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build rental and return station demand tables from OD flow rows."""
    rental = (
        od_flows.groupby(["time_bin", "start_station_id"], as_index=False)["count"]
        .sum()
        .rename(columns={"start_station_id": "station_id", "count": "rental_demand"})
    )
    returns = (
        od_flows.groupby(["time_bin", "end_station_id"], as_index=False)["count"]
        .sum()
        .rename(columns={"end_station_id": "station_id", "count": "return_demand"})
    )
    return rental, returns


def build_district_artifacts(selected_meta: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Create district metadata and arrays aligned to selected station order."""
    district_names = sorted(selected_meta["district_name"].dropna().unique().tolist(), key=SEOUL_DISTRICTS.index)
    district_id_map = {name: idx for idx, name in enumerate(district_names)}
    selected_meta["district_id"] = selected_meta["district_name"].map(district_id_map).astype(int)
    station_district = selected_meta["district_id"].to_numpy(dtype=np.int64)

    district_rows = []
    for district_id, district_name in enumerate(district_names):
        rows = selected_meta.loc[selected_meta["district_id"] == district_id]
        district_rows.append(
            {
                "district_id": int(district_id),
                "district_name": district_name,
                "station_count": int(len(rows)),
                "total_capacity": float(rows["capacity"].sum()),
                "total_rental_demand": float(rows["total_rental_demand"].sum()),
                "total_return_demand": float(rows["total_return_demand"].sum()),
                "latitude": float(rows["latitude"].mean()),
                "longitude": float(rows["longitude"].mean()),
            }
        )

    district_meta = pd.DataFrame(district_rows)
    mask = np.zeros((len(district_names), len(selected_meta)), dtype=bool)
    for station_idx, district_id in enumerate(station_district):
        mask[int(district_id), station_idx] = True
    district_distance = haversine_distance_matrix(district_meta["latitude"], district_meta["longitude"])
    return district_meta, station_district, mask, district_distance


def save_processed_outputs(
    rental_matrix: pd.DataFrame,
    return_matrix: pd.DataFrame,
    selected_meta: pd.DataFrame,
    output_dir: str,
    train_ratio: float,
    summary: dict,
    district_meta: pd.DataFrame | None = None,
    station_district: np.ndarray | None = None,
    district_station_mask: np.ndarray | None = None,
    district_distance_matrix: np.ndarray | None = None,
) -> dict:
    """Save NumPy arrays, metadata CSVs, and preprocessing summary."""
    output_path = ensure_dir(output_dir)
    total_steps = len(rental_matrix)
    if total_steps < 2:
        raise RuntimeError("time step 수가 2보다 작습니다. 더 긴 기간의 OD 데이터를 사용해주세요.")

    train_steps = int(total_steps * train_ratio)
    train_steps = min(max(train_steps, 1), total_steps - 1)

    rental_values = rental_matrix.to_numpy(dtype=np.float32)
    return_values = return_matrix.to_numpy(dtype=np.float32)
    capacity = selected_meta["capacity"].to_numpy(dtype=np.float32)
    distance_matrix = haversine_distance_matrix(selected_meta["latitude"], selected_meta["longitude"])

    np.save(output_path / "rental_train.npy", rental_values[:train_steps])
    np.save(output_path / "return_train.npy", return_values[:train_steps])
    np.save(output_path / "rental_test.npy", rental_values[train_steps:])
    np.save(output_path / "return_test.npy", return_values[train_steps:])
    np.save(output_path / "capacity.npy", capacity)
    np.save(output_path / "distance_matrix.npy", distance_matrix)
    if station_district is not None:
        np.save(output_path / "station_district.npy", station_district.astype(np.int64))
    if district_station_mask is not None:
        np.save(output_path / "district_station_mask.npy", district_station_mask.astype(bool))
    if district_distance_matrix is not None:
        np.save(output_path / "district_distance_matrix.npy", district_distance_matrix.astype(np.float32))

    selected_meta.to_csv(output_path / "station_meta.csv", index=False, encoding="utf-8-sig")
    if district_meta is not None:
        district_meta.to_csv(output_path / "district_meta.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"time_bin": rental_matrix.index[:train_steps]}).to_csv(
        output_path / "time_index_train.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({"time_bin": rental_matrix.index[train_steps:]}).to_csv(
        output_path / "time_index_test.csv", index=False, encoding="utf-8-sig"
    )

    summary.update(
        {
            "selected_station_count": int(len(selected_meta)),
            "selected_station_ids": selected_meta["station_id"].tolist(),
            "train_time_steps": int(train_steps),
            "test_time_steps": int(total_steps - train_steps),
        }
    )
    save_json(output_path / "preprocess_summary.json", summary)
    print(f"[Save] processed outputs 저장 완료: {output_path}")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="따릉이 OD/station 데이터를 DQN 학습용 matrix로 전처리합니다.")
    parser.add_argument("--od-glob", default=DEFAULT_OD_GLOB)
    parser.add_argument("--station-glob", default=DEFAULT_STATION_GLOB)
    parser.add_argument("--freq", default=DEFAULT_FREQ)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--output-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--use-all-stations", action="store_true")
    parser.add_argument("--zone-mode", choices=["station", "district", "kmeans"], default="station")
    parser.add_argument("--district-col", default=None)
    parser.add_argument("--min-station-demand", type=float, default=1.0)
    parser.add_argument("--drop-cross-district-od", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Run the full preprocessing pipeline."""
    args = parse_args(argv)
    args.freq = normalize_freq(args.freq)
    od_paths = sorted(path for path in glob.glob(args.od_glob) if Path(path).is_file())
    station_paths = sorted(path for path in glob.glob(args.station_glob) if Path(path).is_file())

    if not od_paths:
        raise FileNotFoundError(
            f"OD CSV 파일을 찾지 못했습니다: {args.od_glob}\n"
            "data/raw/od/ 폴더에 tpss_bcycl_od_statnhm_YYYYMMDD.csv 파일을 넣어주세요."
        )
    if not station_paths:
        raise FileNotFoundError(
            f"대여소 정보 엑셀 파일을 찾지 못했습니다: {args.station_glob}\n"
            "data/raw/stations/ 폴더에 공공자전거 대여소 정보 .xlsx 또는 .xls 파일을 넣어주세요."
        )

    header_frame, encoding = read_csv_sample(od_paths[0], nrows=0)
    od_columns = detect_od_columns(header_frame.columns)
    print("[Detect] OD columns:", od_columns)
    print(f"[Detect] CSV encoding: {encoding}")

    rental, returns, od_flows, od_station_names, od_stats = aggregate_od_files(od_paths, od_columns, args.freq)
    station_frame, station_columns, station_file, station_header = read_station_excel(station_paths)
    print("[Detect] Station columns:", station_columns)
    station_meta = build_station_meta(station_frame, station_columns)

    selected_ids, selected_meta, match_stats = select_stations(
        rental,
        returns,
        station_meta,
        args.top_n,
        od_station_names,
        use_all_stations=args.use_all_stations,
        min_station_demand=args.min_station_demand,
    )
    dropped_no_district_station_count = 0
    district_source_col = None
    district_meta = None
    station_district = None
    district_station_mask = None
    district_distance_matrix = None
    district_flow_stats = {
        "same_district_od_count": 0,
        "cross_district_od_count": 0,
        "cross_district_od_ratio": 0.0,
    }

    if args.zone_mode == "district":
        selected_meta, district_source_col, dropped_no_district_station_count = attach_station_district(
            selected_meta,
            station_frame,
            station_columns,
            args.district_col,
        )
        if selected_meta.empty:
            raise RuntimeError("자치구가 확인된 선택 대여소가 없습니다.")
        selected_ids = selected_meta["station_id"].tolist()
        district_flows, district_flow_stats = summarize_od_district_flows(
            od_flows,
            selected_meta,
            args.drop_cross_district_od,
        )
        if args.drop_cross_district_od and district_flows is not None:
            rental, returns = demand_from_flows(district_flows)
            rental_totals = rental.groupby("station_id")["rental_demand"].sum()
            return_totals = returns.groupby("station_id")["return_demand"].sum()
            selected_meta["total_rental_demand"] = selected_meta["station_id"].map(rental_totals).fillna(0.0)
            selected_meta["total_return_demand"] = selected_meta["station_id"].map(return_totals).fillna(0.0)
            selected_meta["total_demand"] = selected_meta["total_rental_demand"] + selected_meta["total_return_demand"]
        district_meta, station_district, district_station_mask, district_distance_matrix = build_district_artifacts(
            selected_meta
        )
    elif args.zone_mode == "kmeans":
        raise NotImplementedError("zone-mode=kmeans는 아직 구현되지 않았습니다. station 또는 district를 사용하세요.")

    rental_matrix, return_matrix = build_demand_matrices(rental, returns, selected_ids, args.freq)

    summary = {
        "od_file_count": int(len(od_paths)),
        "station_file": station_file,
        "station_header_row": station_header,
        "freq": args.freq,
        "top_n": int(args.top_n),
        "use_all_stations": bool(args.use_all_stations),
        "zone_mode": args.zone_mode,
        "min_station_demand": float(args.min_station_demand),
        "drop_cross_district_od": bool(args.drop_cross_district_od),
        "detected_columns": {
            "od": od_columns,
            "station": station_columns,
            "district_source_col": district_source_col,
        },
        "removed_rows": int(od_stats["removed_rows"]),
        "total_rows": int(od_stats["total_rows"]),
        "od_station_name_count": int(len(od_station_names)),
        "district_count": int(0 if district_meta is None else len(district_meta)),
        "district_names": [] if district_meta is None else district_meta["district_name"].tolist(),
        "district_station_counts": {}
        if district_meta is None
        else dict(zip(district_meta["district_name"], district_meta["station_count"].astype(int))),
        "dropped_no_district_station_count": int(dropped_no_district_station_count),
        "total_capacity": float(selected_meta["capacity"].sum()),
        "total_rental_demand": float(selected_meta["total_rental_demand"].sum()),
        "total_return_demand": float(selected_meta["total_return_demand"].sum()),
        **district_flow_stats,
        **match_stats,
    }
    return save_processed_outputs(
        rental_matrix,
        return_matrix,
        selected_meta,
        args.output_dir,
        args.train_ratio,
        summary,
        district_meta=district_meta,
        station_district=station_district,
        district_station_mask=district_station_mask,
        district_distance_matrix=district_distance_matrix,
    )


if __name__ == "__main__":
    main()
