"""Deterministic data-quality checks for OpsReport.

The quality score is intentionally simple and auditable. For non-empty data it
is the sum of four components, each bounded by its documented weight:

* required-column coverage: 40 points;
* completeness of present required columns: 25 points;
* numeric validity of present numeric columns: 20 points;
* full-row uniqueness: 15 points.

An empty dataset scores 0. The final score is clamped to the inclusive 0-100
range and rounded to one decimal place.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from .metrics import resolve_column


DEFAULT_REQUIRED_COLUMNS = ("order_id", "revenue")
DEFAULT_NUMERIC_COLUMNS = ("revenue", "quantity", "cost")
QUALITY_SCORE_WEIGHTS = {
    "required_columns": 40.0,
    "completeness": 25.0,
    "numeric_validity": 20.0,
    "uniqueness": 15.0,
}


def _is_blank(series: pd.Series) -> pd.Series:
    """Return a boolean mask for null or whitespace-only values."""

    blank = series.isna()
    if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
        blank = blank | series.astype("string").str.strip().eq("")
    return blank.fillna(True)


def assess_data_quality(
    df: pd.DataFrame,
    required_columns: Iterable[str] = DEFAULT_REQUIRED_COLUMNS,
    numeric_columns: Iterable[str] = DEFAULT_NUMERIC_COLUMNS,
    column_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Assess a dataframe and return a deterministic 0-100 quality report.

    Canonical column names may be supplied through ``column_map`` or resolved
    from the common Spanish/English aliases supported by :func:`resolve_column`.
    Invalid numeric values are reported instead of raising conversion errors.
    """

    required = tuple(dict.fromkeys(required_columns))
    numeric = tuple(dict.fromkeys(numeric_columns))
    issues: list[dict[str, Any]] = []

    if df.empty:
        issues.append(
            {
                "code": "empty_dataset",
                "severity": "error",
                "column": None,
                "count": 0,
                "message": "El archivo no contiene filas para analizar.",
            }
        )
        return {
            "score": 0.0,
            "issues": issues,
            "warnings": [issue["message"] for issue in issues],
            "components": {key: 0.0 for key in QUALITY_SCORE_WEIGHTS},
            "missing_count": 0,
            "missing_percentage": 0.0,
            "duplicate_count": 0,
            "stats": {
                "rows": 0,
                "columns": int(len(df.columns)),
                "duplicate_rows": 0,
                "missing_required_columns": list(required),
                "invalid_numeric_values": 0,
            },
        }

    resolved_required = {
        canonical: resolve_column(df, canonical, column_map) for canonical in required
    }
    missing_required = [name for name, actual in resolved_required.items() if actual is None]
    for canonical in missing_required:
        issues.append(
            {
                "code": "missing_required_column",
                "severity": "error",
                "column": canonical,
                "count": None,
                "message": f"Falta la columna requerida «{canonical}».",
            }
        )

    if required:
        present_ratio = (len(required) - len(missing_required)) / len(required)
        required_score = QUALITY_SCORE_WEIGHTS["required_columns"] * present_ratio
    else:
        required_score = QUALITY_SCORE_WEIGHTS["required_columns"]

    required_cells = 0
    blank_required_cells = 0
    for canonical, actual in resolved_required.items():
        if actual is None:
            continue
        blank_count = int(_is_blank(df[actual]).sum())
        required_cells += len(df)
        blank_required_cells += blank_count
        if blank_count:
            issues.append(
                {
                    "code": "missing_required_values",
                    "severity": "warning",
                    "column": canonical,
                    "count": blank_count,
                    "message": (
                        f"La columna «{canonical}» tiene {blank_count} valor(es) vacío(s)."
                    ),
                }
            )

    if required_cells:
        completeness_ratio = 1.0 - (blank_required_cells / required_cells)
    else:
        completeness_ratio = 0.0 if required else 1.0
    completeness_score = QUALITY_SCORE_WEIGHTS["completeness"] * completeness_ratio

    numeric_inputs = 0
    invalid_numeric_values = 0
    resolved_numeric: set[str] = set()
    for canonical in numeric:
        actual = resolve_column(df, canonical, column_map)
        if actual is None or actual in resolved_numeric:
            continue
        resolved_numeric.add(actual)
        blank_mask = _is_blank(df[actual])
        nonblank = df.loc[~blank_mask, actual]
        numeric_inputs += len(nonblank)
        coerced = pd.to_numeric(nonblank, errors="coerce")
        invalid_count = int(coerced.isna().sum())
        invalid_numeric_values += invalid_count
        if invalid_count:
            issues.append(
                {
                    "code": "invalid_numeric_values",
                    "severity": "warning",
                    "column": canonical,
                    "count": invalid_count,
                    "message": (
                        f"La columna «{canonical}» tiene {invalid_count} valor(es) "
                        "que no se pueden interpretar como número."
                    ),
                }
            )

    numeric_validity_ratio = (
        1.0 - (invalid_numeric_values / numeric_inputs) if numeric_inputs else 1.0
    )
    numeric_score = QUALITY_SCORE_WEIGHTS["numeric_validity"] * numeric_validity_ratio

    duplicate_rows = int(df.duplicated().sum())
    duplicate_ratio = duplicate_rows / len(df)
    uniqueness_score = QUALITY_SCORE_WEIGHTS["uniqueness"] * (1.0 - duplicate_ratio)
    if duplicate_rows:
        issues.append(
            {
                "code": "duplicate_rows",
                "severity": "warning",
                "column": None,
                "count": duplicate_rows,
                "message": f"Se detectaron {duplicate_rows} fila(s) duplicada(s).",
            }
        )

    components = {
        "required_columns": round(required_score, 1),
        "completeness": round(completeness_score, 1),
        "numeric_validity": round(numeric_score, 1),
        "uniqueness": round(uniqueness_score, 1),
    }
    score = round(min(100.0, max(0.0, sum(components.values()))), 1)
    missing_count = int(sum(int(_is_blank(df[column]).sum()) for column in df.columns))
    total_cells = int(df.shape[0] * df.shape[1])
    missing_percentage = (missing_count / total_cells * 100.0) if total_cells else 0.0

    return {
        "score": score,
        "issues": issues,
        "warnings": [issue["message"] for issue in issues],
        "components": components,
        "missing_count": missing_count,
        "missing_percentage": round(missing_percentage, 1),
        "duplicate_count": duplicate_rows,
        "stats": {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "duplicate_rows": duplicate_rows,
            "missing_required_columns": missing_required,
            "invalid_numeric_values": invalid_numeric_values,
        },
    }
