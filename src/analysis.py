"""Deterministic comparative and segment analysis for OpsReport Phase 4."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import pandas as pd

from .metrics import calculate_kpis
from .schema import resolve_column


@dataclass(frozen=True)
class AnalysisConfig:
    """User-visible thresholds for deterministic advanced analysis."""

    sla_hours: float = 24.0
    cancellation_alert_rate: float = 10.0
    trend_deviation_pct: float = 50.0
    trend_window_days: int = 28
    trend_min_history: int = 14
    min_segment_size: int = 3


PERIOD_METRICS = (
    ("revenue", "Ingresos", "currency"),
    ("order_count", "Pedidos", "count"),
    ("units", "Unidades", "count"),
    ("gross_margin", "Margen bruto", "currency"),
    ("gross_margin_rate", "Margen bruto (%)", "percent"),
    ("cancellation_rate", "Cancelación", "percent"),
    ("average_processing_time_hours", "Procesamiento promedio", "hours"),
)

SEGMENT_FIELDS = ("region", "category", "channel", "status")
SEGMENT_LABELS = {
    "region": "Región",
    "category": "Categoría",
    "channel": "Canal",
    "status": "Estado",
}


def build_operational_analysis(
    df: pd.DataFrame,
    column_map: Mapping[str, str | None] | None = None,
    config: AnalysisConfig | None = None,
) -> dict[str, Any]:
    """Build the complete Phase 4 analysis package.

    Results are plain mappings/dataframes so Streamlit and Excel export can reuse
    the same deterministic calculations without becoming analytical authorities.
    """

    resolved_config = config or AnalysisConfig()
    _validate_config(resolved_config)

    period = _period_windows(df, column_map)
    period_metrics = _period_metric_table(period, column_map)
    current_source = period["current_df"] if period.get("available") else df
    segment_summary = _segment_summary(current_source, column_map, resolved_config)
    segment_comparison = _segment_comparison(period, column_map, resolved_config)
    sla = _sla_summary(current_source, column_map, resolved_config)
    trend_anomalies = _time_series_anomalies(df, column_map, resolved_config)

    return {
        "config": asdict(resolved_config),
        "period": {key: value for key, value in period.items() if key not in {"current_df", "previous_df"}},
        "period_metrics": period_metrics,
        "segment_summary": segment_summary,
        "segment_comparison": segment_comparison,
        "sla": sla,
        "trend_anomalies": trend_anomalies,
    }


def _validate_config(config: AnalysisConfig) -> None:
    if not math.isfinite(config.sla_hours) or config.sla_hours <= 0:
        raise ValueError("sla_hours debe ser mayor que cero")
    if not math.isfinite(config.cancellation_alert_rate) or not 0 <= config.cancellation_alert_rate <= 100:
        raise ValueError("cancellation_alert_rate debe estar entre 0 y 100")
    if not math.isfinite(config.trend_deviation_pct) or config.trend_deviation_pct <= 0:
        raise ValueError("trend_deviation_pct debe ser mayor que cero")
    if config.trend_window_days < 2:
        raise ValueError("trend_window_days debe ser al menos 2")
    if config.trend_min_history < 2 or config.trend_min_history > config.trend_window_days:
        raise ValueError("trend_min_history debe estar entre 2 y trend_window_days")
    if config.min_segment_size < 1:
        raise ValueError("min_segment_size debe ser al menos 1")


def _period_windows(
    df: pd.DataFrame,
    column_map: Mapping[str, str | None] | None,
) -> dict[str, Any]:
    date_column = resolve_column(df, "date", column_map)
    empty = {
        "available": False,
        "reason": "No hay una columna de fecha interpretable para comparar períodos.",
        "current_df": df.iloc[0:0].copy(),
        "previous_df": df.iloc[0:0].copy(),
    }
    if date_column is None:
        return empty

    parsed = pd.to_datetime(df[date_column], errors="coerce")
    valid = parsed.dropna()
    if valid.empty:
        return empty

    earliest = valid.min().normalize()
    latest = valid.max().normalize()
    span_days = int((latest - earliest).days)
    if span_days >= 28:
        grain = "month"
        current_start = latest.replace(day=1)
        natural_end = (current_start + pd.offsets.MonthEnd(1)).normalize()
        previous_start = (current_start - pd.DateOffset(months=1)).normalize()
    elif span_days >= 7:
        grain = "week"
        current_start = latest - pd.Timedelta(days=int(latest.weekday()))
        natural_end = current_start + pd.Timedelta(days=6)
        previous_start = current_start - pd.Timedelta(days=7)
    else:
        grain = "day"
        current_start = latest
        natural_end = latest
        previous_start = latest - pd.Timedelta(days=1)

    current_incomplete = latest < natural_end
    if current_incomplete:
        elapsed_days = int((latest - current_start).days)
        previous_end = previous_start + pd.Timedelta(days=elapsed_days)
    else:
        previous_end = current_start - pd.Timedelta(days=1) if grain != "day" else previous_start

    current_mask = parsed.dt.normalize().between(current_start, latest, inclusive="both")
    previous_mask = parsed.dt.normalize().between(previous_start, previous_end, inclusive="both")
    current_df = df.loc[current_mask.fillna(False)].copy()
    previous_df = df.loc[previous_mask.fillna(False)].copy()

    available = not current_df.empty and not previous_df.empty
    if available:
        reason = None
    elif previous_df.empty:
        reason = "No hay registros en el período anterior comparable."
    else:
        reason = "No hay registros en el período actual comparable."

    if grain == "month" and current_incomplete:
        note = (
            "El período actual está incompleto; se compara mes a la fecha contra el mismo "
            "número de días del mes anterior."
        )
    elif grain == "month":
        note = "El último mes observado está completo y se compara contra el mes calendario anterior."
    elif grain == "week" and current_incomplete:
        note = (
            "La semana observada está incompleta; se compara contra los mismos días transcurridos "
            "de la semana anterior."
        )
    elif grain == "week":
        note = "La última semana observada está completa y se compara contra la semana anterior."
    else:
        note = "Se compara el último día observado contra el día calendario anterior."

    return {
        "available": available,
        "reason": reason,
        "date_column": date_column,
        "grain": grain,
        "current_start": current_start,
        "current_end": latest,
        "previous_start": previous_start,
        "previous_end": previous_end,
        "current_label": f"{current_start:%d-%m-%Y} → {latest:%d-%m-%Y}",
        "previous_label": f"{previous_start:%d-%m-%Y} → {previous_end:%d-%m-%Y}",
        "current_incomplete": current_incomplete,
        "note": note,
        "current_rows": int(len(current_df)),
        "previous_rows": int(len(previous_df)),
        "current_df": current_df,
        "previous_df": previous_df,
    }


def _period_metric_table(
    period: Mapping[str, Any],
    column_map: Mapping[str, str | None] | None,
) -> pd.DataFrame:
    columns = [
        "metric",
        "label",
        "unit",
        "current",
        "previous",
        "absolute_change",
        "relative_change_pct",
    ]
    if not period.get("available"):
        return pd.DataFrame(columns=columns)

    current = calculate_kpis(period["current_df"], column_map)
    previous = calculate_kpis(period["previous_df"], column_map)
    rows: list[dict[str, Any]] = []
    for metric, label, unit in PERIOD_METRICS:
        current_value = _as_float(current.get(metric))
        previous_value = _as_float(previous.get(metric))
        if current_value is None and previous_value is None:
            continue
        absolute_change = None
        relative_change = None
        if current_value is not None and previous_value is not None:
            absolute_change = current_value - previous_value
            if previous_value != 0:
                relative_change = (absolute_change / abs(previous_value)) * 100.0
        rows.append(
            {
                "metric": metric,
                "label": label,
                "unit": unit,
                "current": current_value,
                "previous": previous_value,
                "absolute_change": absolute_change,
                "relative_change_pct": relative_change,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _segment_summary(
    df: pd.DataFrame,
    column_map: Mapping[str, str | None] | None,
    config: AnalysisConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for field in SEGMENT_FIELDS:
        segment_column = resolve_column(df, field, column_map)
        if segment_column is None:
            continue
        for segment, group in _groups(df, segment_column):
            row = _segment_metrics(group, column_map, config)
            if field == "status":
                row["cancellation_rate"] = None
                row["high_cancellation"] = False
            row.update(
                {
                    "segment_type": field,
                    "segment_label": SEGMENT_LABELS[field],
                    "segment_column": segment_column,
                    "segment": segment,
                    "eligible_for_highlight": len(group) >= config.min_segment_size,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _segment_comparison(
    period: Mapping[str, Any],
    column_map: Mapping[str, str | None] | None,
    config: AnalysisConfig,
) -> pd.DataFrame:
    if not period.get("available"):
        return pd.DataFrame()

    current_df = period["current_df"]
    previous_df = period["previous_df"]
    rows: list[dict[str, Any]] = []
    for field in SEGMENT_FIELDS:
        current_column = resolve_column(current_df, field, column_map)
        previous_column = resolve_column(previous_df, field, column_map)
        segment_column = current_column or previous_column
        if segment_column is None:
            continue

        values = set(_segment_values(current_df, current_column)) | set(
            _segment_values(previous_df, previous_column)
        )
        for segment in sorted(values, key=str):
            current_group = _segment_slice(current_df, current_column, segment)
            previous_group = _segment_slice(previous_df, previous_column, segment)

            current = _segment_metrics(current_group, column_map, config)
            previous = _segment_metrics(previous_group, column_map, config)
            rows.append(
                {
                    "segment_type": field,
                    "segment_label": SEGMENT_LABELS[field],
                    "segment": segment,
                    "current_rows": int(len(current_group)),
                    "previous_rows": int(len(previous_group)),
                    "eligible_for_highlight": (
                        len(current_group) >= config.min_segment_size
                        and len(previous_group) >= config.min_segment_size
                    ),
                    "revenue_change_pct": _pct_change(current["revenue"], previous["revenue"]),
                    "margin_change_pct": _pct_change(
                        current["gross_margin"], previous["gross_margin"]
                    ),
                    "cancellation_change_pp": (
                        None
                        if field == "status"
                        else _difference(current["cancellation_rate"], previous["cancellation_rate"])
                    ),
                    "processing_change_pct": _pct_change(
                        current["average_processing_time_hours"],
                        previous["average_processing_time_hours"],
                    ),
                    "sla_breach_change_pp": _difference(
                        current["sla_breach_rate"], previous["sla_breach_rate"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _segment_metrics(
    group: pd.DataFrame,
    column_map: Mapping[str, str | None] | None,
    config: AnalysisConfig,
) -> dict[str, Any]:
    kpis = calculate_kpis(group, column_map)
    processing = _processing_series(group, column_map)
    breach_rate = float(processing.gt(config.sla_hours).mean() * 100) if not processing.empty else None
    return {
        "rows": int(len(group)),
        "orders": kpis.get("order_count"),
        "revenue": kpis.get("revenue"),
        "gross_margin": kpis.get("gross_margin"),
        "gross_margin_rate": kpis.get("gross_margin_rate"),
        "cancellation_rate": kpis.get("cancellation_rate"),
        "average_processing_time_hours": kpis.get("average_processing_time_hours"),
        "sla_breach_rate": breach_rate,
        "high_cancellation": (
            kpis.get("cancellation_rate") is not None
            and float(kpis["cancellation_rate"]) >= config.cancellation_alert_rate
        ),
        "sla_risk": breach_rate is not None and breach_rate > 0,
    }


def _sla_summary(
    df: pd.DataFrame,
    column_map: Mapping[str, str | None] | None,
    config: AnalysisConfig,
) -> dict[str, Any]:
    processing = _processing_series(df, column_map)
    if processing.empty:
        return {
            "available": False,
            "threshold_hours": config.sla_hours,
            "reason": "No hay tiempos de procesamiento interpretables.",
        }
    breaches = processing.gt(config.sla_hours)
    return {
        "available": True,
        "threshold_hours": config.sla_hours,
        "observations": int(len(processing)),
        "breach_count": int(breaches.sum()),
        "breach_rate": float(breaches.mean() * 100.0),
        "average_hours": float(processing.mean()),
        "p95_hours": float(processing.quantile(0.95)),
    }


def _time_series_anomalies(
    df: pd.DataFrame,
    column_map: Mapping[str, str | None] | None,
    config: AnalysisConfig,
) -> pd.DataFrame:
    date_column = resolve_column(df, "date", column_map)
    revenue_column = resolve_column(df, "revenue", column_map)
    columns = ["date", "value", "baseline", "deviation_pct", "direction", "metric", "method"]
    if date_column is None or revenue_column is None:
        return pd.DataFrame(columns=columns)

    working = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_column], errors="coerce").dt.normalize(),
            "value": pd.to_numeric(df[revenue_column], errors="coerce"),
        }
    ).dropna()
    if working.empty:
        return pd.DataFrame(columns=columns)

    daily = working.groupby("date", as_index=True)["value"].sum().sort_index().to_frame()
    calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(calendar).rename_axis("date").reset_index()
    history = daily["value"].shift(1)
    rolling = history.rolling(
        window=config.trend_window_days,
        min_periods=config.trend_min_history,
    )
    daily["baseline"] = rolling.median()
    daily["baseline_mad"] = rolling.apply(
        lambda values: (values - values.median()).abs().median(),
        raw=False,
    )
    valid_baseline = daily["baseline"].notna() & daily["baseline"].ne(0)
    daily["deviation_pct"] = ((daily["value"] - daily["baseline"]) / daily["baseline"].abs()) * 100.0
    daily["robust_z"] = (
        0.6745 * (daily["value"] - daily["baseline"]).abs() / daily["baseline_mad"]
    )
    robust_outlier = daily["robust_z"].ge(3.5) | daily["baseline_mad"].eq(0)
    flagged = daily.loc[
        valid_baseline
        & daily["deviation_pct"].abs().ge(config.trend_deviation_pct)
        & robust_outlier
    ].copy()
    if flagged.empty:
        return pd.DataFrame(columns=columns)

    flagged["direction"] = flagged["deviation_pct"].map(lambda value: "high" if value > 0 else "low")
    flagged["metric"] = "revenue_daily"
    flagged["method"] = f"rolling_median_mad_{config.trend_window_days}d"
    return flagged[columns].reset_index(drop=True)


def _processing_series(
    df: pd.DataFrame,
    column_map: Mapping[str, str | None] | None,
) -> pd.Series:
    direct = resolve_column(df, "processing_time_hours", column_map)
    if direct is not None:
        values = pd.to_numeric(df[direct], errors="coerce")
        return values[values >= 0].dropna().astype(float)

    if column_map is not None and "processing_time_hours" in column_map:
        return pd.Series(dtype="float64")

    created = resolve_column(df, "created_at", column_map)
    completed = resolve_column(df, "completed_at", column_map)
    if created is None or completed is None:
        return pd.Series(dtype="float64")
    start = pd.to_datetime(df[created], errors="coerce")
    end = pd.to_datetime(df[completed], errors="coerce")
    hours = (end - start).dt.total_seconds() / 3600.0
    return hours[hours >= 0].dropna().astype(float)


def _groups(df: pd.DataFrame, column: str):
    values = df[column].astype("string").fillna("Sin dato").str.strip().replace("", "Sin dato")
    working = df.assign(_ops_segment=values)
    return working.groupby("_ops_segment", dropna=False, sort=True)


def _segment_values(df: pd.DataFrame, column: str | None) -> list[str]:
    if column is None or df.empty:
        return []
    values = df[column].astype("string").fillna("Sin dato").str.strip().replace("", "Sin dato")
    return [str(value) for value in values.unique()]


def _segment_slice(df: pd.DataFrame, column: str | None, segment: str) -> pd.DataFrame:
    if column is None or df.empty:
        return df.iloc[0:0].copy()
    values = df[column].astype("string").fillna("Sin dato").str.strip().replace("", "Sin dato")
    return df.loc[values.eq(segment)].copy()


def _as_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _pct_change(current: Any, previous: Any) -> float | None:
    current_value = _as_float(current)
    previous_value = _as_float(previous)
    if current_value is None or previous_value is None or previous_value <= 0:
        return None
    return ((current_value - previous_value) / abs(previous_value)) * 100.0


def _difference(current: Any, previous: Any) -> float | None:
    current_value = _as_float(current)
    previous_value = _as_float(previous)
    if current_value is None or previous_value is None:
        return None
    return current_value - previous_value
