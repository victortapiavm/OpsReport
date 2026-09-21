"""Deterministic dataframe profiling helpers used by OpsReport views and exports."""

from __future__ import annotations

from typing import Any

import pandas as pd


def profile_dataframe(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Return a JSON-friendly structural and statistical profile."""
    rows, columns = dataframe.shape
    total_cells = rows * columns
    missing_cells = int(dataframe.isna().sum().sum())
    duplicate_rows = int(dataframe.duplicated().sum())

    return {
        "rows": int(rows),
        "columns": int(columns),
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": _pct(duplicate_rows, rows),
        "missing_cells": missing_cells,
        "missing_pct": _pct(missing_cells, total_cells),
        "numeric_columns": int(len(dataframe.select_dtypes(include="number").columns)),
        "date_columns": detect_date_columns(dataframe),
        "column_types": {str(column): _semantic_type(dataframe[column], str(column)) for column in dataframe.columns},
        "column_profiles": _column_profiles(dataframe),
    }


def missingness_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return one row per column with completeness information."""
    rows = max(len(dataframe), 1)
    missing = dataframe.isna().sum()
    table = pd.DataFrame(
        {
            "column": dataframe.columns.astype(str),
            "dtype": [str(dtype) for dtype in dataframe.dtypes],
            "missing": missing.to_numpy(dtype=int),
            "missing_pct": (missing.to_numpy(dtype=float) / rows * 100).round(2),
            "non_null": dataframe.notna().sum().to_numpy(dtype=int),
            "unique": [int(dataframe[column].nunique(dropna=True)) for column in dataframe.columns],
        }
    )
    return table.sort_values(["missing_pct", "column"], ascending=[False, True]).reset_index(drop=True)


def numeric_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return stable descriptive statistics for numeric columns."""
    numeric = dataframe.select_dtypes(include="number")
    columns = ["column", "count", "missing", "mean", "std", "min", "p25", "median", "p75", "max", "sum"]
    if numeric.empty:
        return pd.DataFrame(columns=columns)

    records: list[dict[str, Any]] = []
    for column in numeric.columns:
        series = pd.to_numeric(numeric[column], errors="coerce")
        records.append(
            {
                "column": str(column),
                "count": int(series.count()),
                "missing": int(series.isna().sum()),
                "mean": _finite_or_none(series.mean()),
                "std": _finite_or_none(series.std()),
                "min": _finite_or_none(series.min()),
                "p25": _finite_or_none(series.quantile(0.25)),
                "median": _finite_or_none(series.median()),
                "p75": _finite_or_none(series.quantile(0.75)),
                "max": _finite_or_none(series.max()),
                "sum": _finite_or_none(series.sum()),
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def categorical_summary(dataframe: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Summarize low-dimensional non-numeric columns and their leading values."""
    columns = ["column", "unique", "missing", "top_value", "top_count", "top_pct", "top_values"]
    candidates = dataframe.select_dtypes(exclude="number")
    records: list[dict[str, Any]] = []
    for column in candidates.columns:
        series = candidates[column]
        counts = series.value_counts(dropna=True)
        top_value = str(counts.index[0]) if not counts.empty else None
        top_count = int(counts.iloc[0]) if not counts.empty else 0
        non_null = int(series.notna().sum())
        leaders = "; ".join(f"{value}: {int(count)}" for value, count in counts.head(top_n).items())
        records.append(
            {
                "column": str(column),
                "unique": int(series.nunique(dropna=True)),
                "missing": int(series.isna().sum()),
                "top_value": top_value,
                "top_count": top_count,
                "top_pct": _pct(top_count, non_null),
                "top_values": leaders,
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def detect_date_columns(dataframe: pd.DataFrame) -> list[str]:
    """Detect datetime columns plus strongly date-named parseable text columns."""
    result: list[str] = []
    hints = ("date", "fecha", "day", "día", "dia", "timestamp", "created", "updated")
    for column in dataframe.columns:
        series = dataframe[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            result.append(str(column))
            continue
        label = str(column).lower()
        if not any(hint in label for hint in hints) or series.dropna().empty:
            continue
        parsed = pd.to_datetime(series.dropna(), errors="coerce")
        if float(parsed.notna().mean()) >= 0.8:
            result.append(str(column))
    return result


def _column_profiles(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    rows = max(len(dataframe), 1)
    profiles: list[dict[str, Any]] = []
    for column in dataframe.columns:
        series = dataframe[column]
        missing = int(series.isna().sum())
        profiles.append(
            {
                "name": str(column),
                "dtype": str(series.dtype),
                "semantic_type": _semantic_type(series, str(column)),
                "non_null": int(series.notna().sum()),
                "missing": missing,
                "missing_pct": round(missing / rows * 100, 2),
                "unique": int(series.nunique(dropna=True)),
            }
        )
    return profiles


def _semantic_type(series: pd.Series, column_name: str) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"

    label = column_name.lower()
    if any(hint in label for hint in ("date", "fecha", "timestamp", "created", "updated")):
        non_null = series.dropna()
        if not non_null.empty:
            parsed = pd.to_datetime(non_null, errors="coerce")
            if float(parsed.notna().mean()) >= 0.8:
                return "date"

    unique = int(series.nunique(dropna=True))
    threshold = max(20, int(len(series) * 0.05))
    return "categorical" if unique <= threshold else "text"


def _pct(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / float(denominator) * 100, 2) if denominator else 0.0


def _finite_or_none(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)
