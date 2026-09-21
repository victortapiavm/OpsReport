"""Business KPI calculations for OpsReport.

All calculations live outside the Streamlit layer so the same behavior can be
tested and reused by exports or narrative generation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from .schema import COLUMN_ALIASES, cancelled_status_mask, resolve_column

def _numeric_series(df: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _safe_sum(series: pd.Series) -> float | None:
    valid = series.dropna()
    return float(valid.sum()) if not valid.empty else None


def _processing_time_hours(
    df: pd.DataFrame,
    column_map: Mapping[str, str | None] | None,
) -> float | None:
    direct_column = resolve_column(df, "processing_time_hours", column_map)
    if direct_column is not None:
        values = _numeric_series(df, direct_column)
        values = values[values >= 0].dropna()
        return float(values.mean()) if not values.empty else None

    if column_map is not None and "processing_time_hours" in column_map:
        return None

    created_column = resolve_column(df, "created_at", column_map)
    completed_column = resolve_column(df, "completed_at", column_map)
    if created_column is None or completed_column is None:
        return None

    created = pd.to_datetime(df[created_column], errors="coerce")
    completed = pd.to_datetime(df[completed_column], errors="coerce")
    durations = (completed - created).dt.total_seconds() / 3600.0
    durations = durations[durations >= 0].dropna()
    return float(durations.mean()) if not durations.empty else None


def calculate_kpis(
    df: pd.DataFrame,
    column_map: Mapping[str, str | None] | None = None,
) -> dict[str, float | int | None]:
    """Calculate the core operations KPIs without raising on imperfect data.

    Missing fields produce ``None`` for dependent KPIs. Numeric text is coerced
    with invalid cells ignored; when a present numeric column has no valid
    values, its KPI is ``None``. Rates are returned as percentages from 0-100.
    """

    revenue_column = resolve_column(df, "revenue", column_map)
    order_column = resolve_column(df, "order_id", column_map)
    quantity_column = resolve_column(df, "quantity", column_map)
    cost_column = resolve_column(df, "cost", column_map)
    status_column = resolve_column(df, "status", column_map)

    revenue = _safe_sum(_numeric_series(df, revenue_column))
    units = _safe_sum(_numeric_series(df, quantity_column))
    cost = _safe_sum(_numeric_series(df, cost_column))

    order_count: int | None = None
    if order_column is not None:
        order_values = df[order_column].dropna()
        if pd.api.types.is_object_dtype(order_values.dtype) or pd.api.types.is_string_dtype(
            order_values.dtype
        ):
            order_values = order_values[order_values.astype("string").str.strip().ne("")]
        order_count = int(order_values.nunique())

    average_order_value = (
        float(revenue / order_count)
        if revenue is not None and order_count is not None and order_count > 0
        else None
    )

    gross_margin = None
    gross_margin_rate = None
    if revenue is not None and cost is not None:
        gross_margin = float(revenue - cost)
        if revenue != 0:
            gross_margin_rate = float((gross_margin / revenue) * 100.0)

    cancellation_rate = None
    if status_column is not None:
        statuses = df[status_column].dropna().astype("string").str.strip()
        statuses = statuses[statuses.ne("")]
        if not statuses.empty:
            cancelled = cancelled_status_mask(statuses)
            cancellation_rate = float(cancelled.mean() * 100.0)

    average_processing_time = _processing_time_hours(df, column_map)

    return {
        "row_count": int(len(df)),
        "order_count": order_count,
        "revenue": revenue,
        "average_order_value": average_order_value,
        "units": units,
        "gross_margin": gross_margin,
        "gross_margin_rate": gross_margin_rate,
        "cancellation_rate": cancellation_rate,
        "average_processing_time_hours": average_processing_time,
        "average_processing_time": average_processing_time,
    }
