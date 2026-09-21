"""Deterministic Spanish management narrative with an optional enhancement hook."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import pandas as pd

from src.profiling import detect_date_columns, profile_dataframe


class NarrativeEnhancer(Protocol):
    """Optional extension point for a future LLM or rules-based rewriter."""

    def enhance(self, base_text: str, context: Mapping[str, Any]) -> str:
        """Return an enhanced version of a complete deterministic narrative."""


@dataclass(frozen=True)
class NarrativeSignals:
    row_count: int
    duplicate_rows: int
    missing_pct: float
    period: str | None = None
    cancellation_rate: float | None = None
    average_processing_hours: float | None = None
    p95_processing_hours: float | None = None
    total_revenue: float | None = None
    poorest_segment: str | None = None
    anomaly_count: int | None = None


def generate_management_narrative(
    dataframe: pd.DataFrame,
    profile: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | pd.DataFrame | None = None,
    anomalies: pd.DataFrame | list[Any] | tuple[Any, ...] | None = None,
    enhancer: NarrativeEnhancer | None = None,
) -> str:
    """Create a concise, repeatable Spanish executive narrative.

    The base narrative never calls an external service. An optional ``enhancer``
    receives both the finished text and structured signals, so an LLM can be
    introduced later without changing analysis or export contracts.
    """
    resolved_profile = dict(profile or profile_dataframe(dataframe))
    signals = _extract_signals(dataframe, resolved_profile, metrics, anomalies)
    paragraphs = _render(signals)
    base_text = "\n\n".join(paragraphs)

    if enhancer is None:
        return base_text
    context = {"signals": signals.__dict__, "profile": resolved_profile}
    return enhancer.enhance(base_text, context)


build_management_narrative = generate_management_narrative


def _extract_signals(
    dataframe: pd.DataFrame,
    profile: Mapping[str, Any],
    metrics: Mapping[str, Any] | pd.DataFrame | None,
    anomalies: pd.DataFrame | list[Any] | tuple[Any, ...] | None,
) -> NarrativeSignals:
    status_col = _find_column(dataframe, ("status", "estado"))
    processing_col = _find_column(
        dataframe,
        ("processing_hours", "processing_time_hours", "processing_time", "tiempo_proceso_horas", "tiempo_proceso", "lead_time_hours"),
    )
    revenue_col = _find_column(dataframe, ("revenue_clp", "revenue", "ingresos_clp", "ingresos", "sales", "ventas"))
    segment_col = _find_column(dataframe, ("region", "región", "category", "categoria", "categoría"))

    cancellation_rate = _metric_value(metrics, ("cancellation_rate", "cancel_rate", "tasa_cancelacion"))
    if cancellation_rate is None and status_col:
        status = dataframe[status_col].astype("string").str.lower().str.strip()
        cancelled = status.str.contains(r"cancel|anulad", regex=True, na=False)
        cancellation_rate = float(cancelled.mean() * 100)

    average_processing = _metric_value(metrics, ("average_processing_hours", "avg_processing_time", "processing_mean"))
    p95_processing = _metric_value(metrics, ("p95_processing_hours", "processing_p95"))
    if processing_col:
        numeric = pd.to_numeric(dataframe[processing_col], errors="coerce")
        if average_processing is None and numeric.notna().any():
            average_processing = float(numeric.mean())
        if p95_processing is None and numeric.notna().any():
            p95_processing = float(numeric.quantile(0.95))

    total_revenue = _metric_value(metrics, ("total_revenue", "revenue_total", "ingresos_totales"))
    if total_revenue is None and revenue_col:
        revenue = pd.to_numeric(dataframe[revenue_col], errors="coerce")
        if revenue.notna().any():
            total_revenue = float(revenue.sum())

    poorest_segment = _poorest_segment(dataframe, segment_col, processing_col, status_col)
    period = _period_text(dataframe)
    anomaly_count = _anomaly_count(anomalies)

    return NarrativeSignals(
        row_count=int(profile.get("rows", len(dataframe))),
        duplicate_rows=int(profile.get("duplicate_rows", dataframe.duplicated().sum())),
        missing_pct=float(profile.get("missing_pct", 0.0)),
        period=period,
        cancellation_rate=cancellation_rate,
        average_processing_hours=average_processing,
        p95_processing_hours=p95_processing,
        total_revenue=total_revenue,
        poorest_segment=poorest_segment,
        anomaly_count=anomaly_count,
    )


def _render(signals: NarrativeSignals) -> list[str]:
    scope = f"Se analizaron {_format_int(signals.row_count)} registros"
    if signals.period:
        scope += f" correspondientes al período {signals.period}"
    scope += "."

    performance_parts: list[str] = []
    if signals.cancellation_rate is not None:
        performance_parts.append(f"la tasa de cancelación fue de {signals.cancellation_rate:.1f}%")
    if signals.average_processing_hours is not None:
        text = f"el tiempo medio de proceso alcanzó {signals.average_processing_hours:.1f} h"
        if signals.p95_processing_hours is not None:
            text += f" (p95: {signals.p95_processing_hours:.1f} h)"
        performance_parts.append(text)
    if signals.total_revenue is not None:
        performance_parts.append(f"los ingresos registrados sumaron ${_format_int(signals.total_revenue)}")
    performance = "Desempeño operativo: " + "; ".join(performance_parts) + "." if performance_parts else "Desempeño operativo: no hay métricas estándar suficientes para sintetizar volumen, tiempos o ingresos."

    quality = (
        f"Calidad de datos: {signals.missing_pct:.1f}% de las celdas están vacías y "
        f"se detectaron {_format_int(signals.duplicate_rows)} filas duplicadas."
    )
    if signals.anomaly_count is not None:
        quality += f" El análisis marcó {_format_int(signals.anomaly_count)} observaciones anómalas."

    priorities: list[str] = []
    if signals.cancellation_rate is not None and signals.cancellation_rate >= 10:
        priorities.append("revisar las causas de cancelación")
    if signals.poorest_segment:
        priorities.append(f"profundizar el desempeño de {signals.poorest_segment}")
    if signals.missing_pct >= 5:
        priorities.append("mejorar la completitud de los campos de origen")
    if signals.duplicate_rows:
        priorities.append("depurar duplicados antes de usar los datos para seguimiento periódico")
    if not priorities:
        priorities.append("mantener el seguimiento de tendencia y revisar los principales segmentos ante desvíos")
    priority_text = "Prioridades sugeridas: " + "; ".join(priorities) + "."
    return [scope, performance, quality, priority_text]


def _find_column(dataframe: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    normalized = {_normalize_name(column): str(column) for column in dataframe.columns}
    for candidate in candidates:
        match = normalized.get(_normalize_name(candidate))
        if match is not None:
            return match
    return None


def _normalize_name(value: Any) -> str:
    translation = str.maketrans("áéíóúüñ", "aeiouun")
    return str(value).strip().lower().translate(translation).replace(" ", "_")


def _metric_value(metrics: Mapping[str, Any] | pd.DataFrame | None, keys: tuple[str, ...]) -> float | None:
    if not isinstance(metrics, Mapping):
        return None
    normalized = {_normalize_name(key): value for key, value in metrics.items()}
    for key in keys:
        value = normalized.get(_normalize_name(key))
        try:
            if value is not None and not pd.isna(value):
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _period_text(dataframe: pd.DataFrame) -> str | None:
    date_columns = detect_date_columns(dataframe)
    if not date_columns:
        return None
    parsed = pd.to_datetime(dataframe[date_columns[0]], errors="coerce").dropna()
    if parsed.empty:
        return None
    return f"{parsed.min():%d-%m-%Y} a {parsed.max():%d-%m-%Y}"


def _poorest_segment(
    dataframe: pd.DataFrame,
    segment_col: str | None,
    processing_col: str | None,
    status_col: str | None,
) -> str | None:
    if not segment_col:
        return None
    valid = dataframe.dropna(subset=[segment_col])
    if valid.empty or valid[segment_col].nunique() < 2:
        return None

    if processing_col:
        working = valid.assign(_metric=pd.to_numeric(valid[processing_col], errors="coerce")).dropna(subset=["_metric"])
        grouped = working.groupby(segment_col, dropna=True)["_metric"].agg(["mean", "count"])
        grouped = grouped[grouped["count"] >= max(3, len(dataframe) * 0.01)]
        if len(grouped) >= 2:
            worst = grouped["mean"].idxmax()
            worst_mean = float(grouped.loc[worst, "mean"])
            overall = float(working["_metric"].mean())
            if overall > 0 and worst_mean >= overall * 1.15:
                return f"{segment_col} = {worst}"

    if status_col:
        cancelled = valid[status_col].astype("string").str.lower().str.contains(r"cancel|anulad", regex=True, na=False)
        rates = cancelled.groupby(valid[segment_col]).mean()
        if len(rates) >= 2 and float(rates.max()) >= max(float(cancelled.mean()) * 1.25, 0.08):
            return f"{segment_col} = {rates.idxmax()}"
    return None


def _anomaly_count(anomalies: pd.DataFrame | list[Any] | tuple[Any, ...] | None) -> int | None:
    if anomalies is None:
        return None
    return int(len(anomalies))


def _format_int(value: int | float) -> str:
    return f"{int(round(value)):,}".replace(",", ".")
