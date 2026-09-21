"""Deterministic robust anomaly detection for business measures."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from .metrics import resolve_column


DEFAULT_ANOMALY_COLUMNS = ("revenue", "quantity", "cost", "processing_time_hours")
DISPLAY_NAMES = {
    "revenue": "ingresos",
    "quantity": "unidades",
    "cost": "costos",
    "processing_time_hours": "tiempo de procesamiento",
}


def _format_value(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value):,}".replace(",", ".")
    return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _outlier_mask(values: pd.Series) -> tuple[pd.Series, str]:
    """Return a robust outlier mask and the method used.

    Modified z-score based on median absolute deviation (MAD) is preferred.
    Tukey's 1.5*IQR fences are used when MAD is zero. If both dispersion
    measures are zero, any deviation from the constant median is considered an
    anomaly because the observed baseline otherwise has no variation.
    """

    median = float(values.median())
    absolute_deviation = (values - median).abs()
    mad = float(absolute_deviation.median())

    if mad > 0:
        modified_z = 0.6745 * absolute_deviation / mad
        return modified_z.gt(3.5), "mad"

    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    if iqr > 0:
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return values.lt(lower) | values.gt(upper), "iqr"

    return values.ne(median), "constant_baseline"


def detect_anomalies(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    column_map: Mapping[str, str] | None = None,
    min_samples: int = 4,
) -> list[dict[str, Any]]:
    """Detect numeric outliers and return understandable Spanish findings.

    The detector is deterministic and ignores missing/nonnumeric values. By
    default it inspects common operational measures only, avoiding accidental
    treatment of numeric identifiers as continuous measures. A requested entry
    in ``columns`` may be either a canonical field or an actual dataframe
    column name.
    """

    if min_samples < 3:
        raise ValueError("min_samples debe ser al menos 3")

    requested = tuple(columns) if columns is not None else DEFAULT_ANOMALY_COLUMNS
    findings: list[dict[str, Any]] = []
    seen_actual_columns: set[str] = set()

    for requested_name in requested:
        actual_column = resolve_column(df, requested_name, column_map)
        if actual_column is None and requested_name in df.columns:
            actual_column = requested_name
        if actual_column is None or actual_column in seen_actual_columns:
            continue
        seen_actual_columns.add(actual_column)

        numeric = pd.to_numeric(df[actual_column], errors="coerce")
        valid_frame = pd.DataFrame(
            {"value": numeric.to_numpy(), "row_position": range(len(df))},
            index=df.index,
        ).dropna(subset=["value"])
        valid = valid_frame["value"]
        if len(valid) < min_samples:
            continue

        mask, method = _outlier_mask(valid)
        if not mask.any():
            continue

        median = float(valid.median())
        canonical_name = requested_name if requested_name in DISPLAY_NAMES else actual_column
        display_name = DISPLAY_NAMES.get(canonical_name, str(actual_column))
        flagged = valid_frame.loc[mask]
        for index, row in flagged.iterrows():
            numeric_value = float(row["value"])
            direction = "high" if numeric_value > median else "low"
            direction_text = "por encima" if direction == "high" else "por debajo"
            findings.append(
                {
                    "row_index": index,
                    "row_position": int(row["row_position"]),
                    "column": str(actual_column),
                    "canonical_column": str(canonical_name),
                    "value": numeric_value,
                    "median": median,
                    "direction": direction,
                    "method": method,
                    "severity": "warning",
                    "message": (
                        f"Valor atípico en {display_name}: {_format_value(numeric_value)} "
                        f"está {direction_text} del patrón habitual "
                        f"(mediana {_format_value(median)})."
                    ),
                }
            )

    return findings
