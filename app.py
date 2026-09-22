from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis import AnalysisConfig, build_operational_analysis
from src.anomalies import detect_anomalies
from src.exports import build_excel_export
from src.ingestion import load_data
from src.metrics import calculate_kpis
from src.narrative import generate_management_narrative
from src.profiling import profile_dataframe
from src.schema import (
    FIELD_LABELS,
    SEMANTIC_FIELDS,
    ColumnInference,
    cancelled_status_mask,
    infer_schema,
    resolve_column,
    suggested_mapping,
    validate_column_mapping,
)
from src.validation import assess_data_quality


APP_ROOT = Path(__file__).parent
SAMPLE_PATH = APP_ROOT / "data" / "sample_operations.csv"

st.set_page_config(
    page_title="OpsReport · Análisis operativo",
    page_icon="▦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ops-ink: #16211d;
            --ops-muted: #62706a;
            --ops-border: #dfe6e2;
            --ops-soft: #f5f8f6;
            --ops-accent: #176b52;
        }
        .stApp {
            background: linear-gradient(180deg, #fbfcfb 0%, #ffffff 280px);
            color: var(--ops-ink);
        }
        .block-container {
            max-width: 1180px;
            padding-top: 1.7rem;
            padding-bottom: 4rem;
        }
        h1, h2, h3 {
            letter-spacing: -0.025em;
        }
        h1 {
            font-size: clamp(2.45rem, 4.2vw, 3.75rem) !important;
            line-height: 1.04 !important;
            margin-bottom: 0.9rem !important;
            max-width: 930px;
        }
        h2 {
            margin-top: 2.4rem !important;
            padding-top: 0.25rem;
            font-size: 1.55rem !important;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--ops-border);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            box-shadow: 0 6px 22px rgba(19, 54, 43, 0.035);
        }
        [data-testid="stMetricLabel"] {
            color: var(--ops-muted);
        }
        .ops-eyebrow {
            color: var(--ops-accent);
            font-size: 0.78rem;
            font-weight: 750;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }
        .ops-lead {
            color: var(--ops-muted);
            font-size: 1.08rem;
            line-height: 1.65;
            max-width: 790px;
            margin-bottom: 1rem;
        }
        .ops-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 0.2rem 0 1.7rem;
        }
        .ops-meta span {
            border: 1px solid var(--ops-border);
            background: rgba(255,255,255,0.72);
            color: #526058;
            border-radius: 999px;
            padding: 0.3rem 0.65rem;
            font-size: 0.78rem;
            font-weight: 600;
        }
        .ops-file {
            display: inline-flex;
            gap: 0.55rem;
            align-items: center;
            background: var(--ops-soft);
            border: 1px solid var(--ops-border);
            border-radius: 999px;
            padding: 0.45rem 0.8rem;
            color: #425048;
            font-size: 0.88rem;
            margin: 0.35rem 0 1rem;
        }
        .ops-summary {
            background: #f4f8f6;
            border: 1px solid #dbe8e1;
            border-left: 4px solid var(--ops-accent);
            border-radius: 10px;
            padding: 1.05rem 1.2rem;
            line-height: 1.65;
        }
        .ops-note {
            color: var(--ops-muted);
            font-size: 0.86rem;
        }
        div[data-testid="stButton"] > button,
        div[data-testid="stDownloadButton"] > button {
            border-radius: 10px;
            font-weight: 650;
        }
        div[data-testid="stFileUploader"] {
            border-radius: 12px;
        }
        [data-testid="stAppDeployButton"],
        #MainMenu {
            display: none;
        }
        [data-testid="stAlert"] {
            border-radius: 10px;
        }
        @media (max-width: 780px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            h1 {
                font-size: 2.35rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_number(value: Any, decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "No disponible"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if decimals:
        return f"{number:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{number:,.0f}".replace(",", ".")


def _format_currency(value: Any) -> str:
    if value is None or pd.isna(value):
        return "No disponible"
    return "$" + _format_number(value)


def _format_percent(value: Any) -> str:
    if value is None or pd.isna(value):
        return "No disponible"
    number = float(value)
    if abs(number) <= 1:
        number *= 100
    return f"{number:.1f}%".replace(".", ",")


def _get(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _date_range(
    df: pd.DataFrame,
    column_map: dict[str, str | None] | None = None,
) -> tuple[str, str] | None:
    date_column = resolve_column(df, "date", column_map)
    if not date_column:
        return None
    parsed = pd.to_datetime(df[date_column], errors="coerce")
    parsed = parsed.dropna()
    if parsed.empty:
        return None
    return parsed.min().strftime("%d-%m-%Y"), parsed.max().strftime("%d-%m-%Y")


def _charts(
    df: pd.DataFrame,
    column_map: dict[str, str | None] | None = None,
) -> list[tuple[str, Any]]:
    charts: list[tuple[str, Any]] = []
    date_col = resolve_column(df, "date", column_map)
    revenue_col = resolve_column(df, "revenue", column_map)
    order_col = resolve_column(df, "order_id", column_map)
    region_col = resolve_column(df, "region", column_map)
    status_col = resolve_column(df, "status", column_map)
    category_col = resolve_column(df, "category", column_map)
    processing_col = resolve_column(df, "processing_time_hours", column_map)

    if date_col and revenue_col:
        tmp = df[[date_col, revenue_col]].copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp[revenue_col] = pd.to_numeric(tmp[revenue_col], errors="coerce")
        daily = tmp.dropna().groupby(date_col, as_index=False)[revenue_col].sum()
        if not daily.empty:
            fig = px.line(
                daily,
                x=date_col,
                y=revenue_col,
                markers=True,
                labels={date_col: "Fecha", revenue_col: "Ingresos"},
            )
            fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified")
            charts.append(("Ingresos a lo largo del tiempo", fig))

    if region_col:
        if order_col:
            grouped = (
                df.groupby(region_col, dropna=False)[order_col]
                .nunique()
                .sort_values(ascending=False)
                .reset_index(name="orders")
            )
        else:
            grouped = df.groupby(region_col, dropna=False).size().reset_index(name="orders")
        grouped[region_col] = grouped[region_col].fillna("Sin región").astype(str)
        fig = px.bar(
            grouped,
            x=region_col,
            y="orders",
            labels={region_col: "Región", "orders": "Pedidos"},
        )
        fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
        charts.append(("Pedidos por región", fig))

    if status_col:
        grouped = df[status_col].fillna("Sin estado").astype(str).value_counts().reset_index()
        grouped.columns = ["status", "count"]
        fig = px.bar(
            grouped,
            x="count",
            y="status",
            orientation="h",
            labels={"status": "Estado", "count": "Registros"},
        )
        fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
        charts.append(("Distribución por estado", fig))

    if category_col and revenue_col:
        tmp = df[[category_col, revenue_col]].copy()
        tmp[revenue_col] = pd.to_numeric(tmp[revenue_col], errors="coerce")
        grouped = (
            tmp.dropna(subset=[revenue_col])
            .groupby(category_col, dropna=False)[revenue_col]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        grouped[category_col] = grouped[category_col].fillna("Sin categoría").astype(str)
        fig = px.bar(
            grouped,
            x=category_col,
            y=revenue_col,
            labels={category_col: "Categoría", revenue_col: "Ingresos"},
        )
        fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
        charts.append(("Ingresos por categoría", fig))

    if processing_col:
        processing = pd.to_numeric(df[processing_col], errors="coerce").dropna()
        if not processing.empty:
            fig = px.histogram(
                x=processing,
                nbins=35,
                labels={"x": "Horas de procesamiento", "count": "Registros"},
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title="Horas de procesamiento",
                yaxis_title="Registros",
            )
            charts.append(("Distribución del tiempo de procesamiento", fig))

    if region_col and status_col:
        tmp = df[[region_col, status_col]].copy()
        tmp["_cancelled"] = cancelled_status_mask(tmp[status_col])
        grouped = tmp.groupby(region_col, dropna=False)["_cancelled"].mean().mul(100).reset_index()
        grouped[region_col] = grouped[region_col].fillna("Sin región").astype(str)
        fig = px.bar(
            grouped,
            x=region_col,
            y="_cancelled",
            labels={region_col: "Región", "_cancelled": "Tasa de cancelación (%)"},
        )
        fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
        charts.append(("Cancelaciones por región", fig))

    for _, fig in charts:
        fig.update_layout(
            template="plotly_white",
            height=360,
            font=dict(size=13),
            margin=dict(l=12, r=12, t=18, b=12),
        )
    return charts


def _render_schema_mapping(
    df: pd.DataFrame,
    filename: str | None,
) -> tuple[dict[str, str | None], dict[str, ColumnInference]]:
    """Render explainable schema proposals and return the active column mapping."""

    inferences = infer_schema(df)
    automatic = suggested_mapping(inferences)
    signature = f"{filename or 'dataset'}|{'|'.join(str(column) for column in df.columns)}"
    mapping_store = st.session_state.setdefault("schema_mappings", {})
    saved = mapping_store.get(signature)
    active = {
        field: (
            saved.get(field)
            if isinstance(saved, dict) and field in saved
            else automatic.get(field)
        )
        for field in SEMANTIC_FIELDS
    }

    high_count = sum(inference.confidence == "high" for inference in inferences.values())
    needs_review = any(inference.confidence != "high" for inference in inferences.values())
    title = f"Mapeo de columnas · {high_count}/{len(SEMANTIC_FIELDS)} roles detectados automáticamente"

    with st.expander(title, expanded=needs_review and saved is None):
        st.caption(
            "OpsReport reconoce encabezados con reglas transparentes. Los mapeos de confianza alta "
            "se aplican automáticamente; las sugerencias dudosas requieren confirmación. Puedes dejar "
            "cualquier rol sin mapear y el análisis continuará con lo disponible."
        )

        confidence_labels = {
            "high": "Alta · automática",
            "medium": "Media · revisar",
            "low": "Baja · revisar",
            "none": "Sin coincidencia",
        }
        inference_rows = [
            {
                "Rol": FIELD_LABELS[field],
                "Sugerencia": inference.column or "—",
                "Confianza": confidence_labels[inference.confidence],
                "Evidencia": inference.reason,
            }
            for field, inference in inferences.items()
        ]
        st.dataframe(pd.DataFrame(inference_rows), width="stretch", hide_index=True)

        if saved is None and needs_review:
            st.info(
                "Las sugerencias de confianza media o baja no afectan los cálculos hasta que confirmes "
                "el mapeo."
            )

        form_key = f"schema_mapping_form::{signature}"
        with st.form(form_key):
            st.markdown("**Revisar o corregir mapeo**")
            proposed: dict[str, str | None] = {}
            form_columns = st.columns(3)
            source_columns = [str(column) for column in df.columns]
            options = ["— No mapear —", *source_columns]
            for index, field in enumerate(SEMANTIC_FIELDS):
                inference = inferences[field]
                if isinstance(saved, dict):
                    default_column = active.get(field)
                elif inference.confidence in {"high", "medium"} and inference.column:
                    default_column = inference.column
                else:
                    default_column = None
                default_value = default_column if default_column in source_columns else "— No mapear —"
                option_index = options.index(default_value) if default_value in options else 0
                with form_columns[index % 3]:
                    selection = st.selectbox(
                        FIELD_LABELS[field],
                        options,
                        index=option_index,
                        help=(
                            f"Confianza: {confidence_labels[inference.confidence]}. "
                            f"{inference.reason}"
                        ),
                    )
                proposed[field] = None if selection == "— No mapear —" else selection

            submitted = st.form_submit_button("Aplicar mapeo", type="primary")

        if submitted:
            errors = validate_column_mapping(df, proposed)
            if errors:
                for error in errors:
                    st.error(error)
            else:
                mapping_store[signature] = dict(proposed)
                active = dict(proposed)
                st.success("Mapeo aplicado a este archivo.")

        mapped_count = sum(bool(column) for column in active.values())
        st.caption(f"Mapeo activo: {mapped_count} de {len(SEMANTIC_FIELDS)} roles semánticos.")

    return active, inferences


def _render_quality(quality: dict[str, Any], profile: dict[str, Any]) -> None:
    score = _get(quality, "score", "quality_score", default=0)
    stats = quality.get("stats", {}) if isinstance(quality.get("stats"), dict) else {}
    missing_count = _get(
        quality,
        "missing_count",
        "missing_values",
        "total_missing",
        default=profile.get("missing_cells", 0),
    )
    missing_pct = _get(
        quality,
        "missing_percentage",
        "missing_percent",
        "missing_pct",
        default=profile.get("missing_pct", 0),
    )
    duplicates = stats.get("duplicate_rows", profile.get("duplicate_rows", 0))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Indicador de calidad", f"{float(score):.0f}/100")
    c2.metric("Valores faltantes", _format_number(missing_count))
    c3.metric("Porcentaje faltante", _format_percent(missing_pct))
    c4.metric("Filas duplicadas", _format_number(duplicates))

    st.caption(
        "Indicador diagnóstico práctico y determinístico; no representa una certificación "
        "estadística de calidad."
    )

    warnings = _get(quality, "warnings", "issues", default=[]) or []
    structured_issues = quality.get("issues", []) or []
    messages: list[str] = []
    if int(missing_count):
        messages.append(f"{int(missing_count)} valores faltantes detectados.")
    for warning in warnings:
        if isinstance(warning, dict):
            message = warning.get("message")
            if message:
                messages.append(str(message))
        else:
            messages.append(str(warning))
    if messages:
        summary_bits: list[str] = []
        if int(missing_count):
            summary_bits.append(f"{int(missing_count)} valores faltantes")
        if int(duplicates):
            summary_bits.append(f"{int(duplicates)} filas duplicadas")
        invalid_numeric = int(stats.get("invalid_numeric_values", 0) or 0)
        if invalid_numeric:
            summary_bits.append(f"{invalid_numeric} valores numéricos no interpretables")
        invalid_dates = int(stats.get("invalid_date_values", 0) or 0)
        if invalid_dates:
            summary_bits.append(f"{invalid_dates} fechas no interpretables")
        if summary_bits:
            st.warning(" · ".join(summary_bits))
        with st.expander("Ver detalle de advertencias", expanded=False):
            for message in dict.fromkeys(messages):
                st.write(f"• {message}")
    else:
        st.success("No se detectaron advertencias relevantes con las reglas actuales.")

    type_data = profile.get("column_profiles")
    inconsistencies = [
        issue.get("message")
        for issue in structured_issues
        if isinstance(issue, dict)
        and issue.get("code") in {"invalid_numeric_values", "invalid_date_values"}
    ]
    if type_data:
        type_df = pd.DataFrame(type_data)
        type_df = type_df.rename(
            columns={
                "name": "Columna",
                "dtype": "Dtype",
                "semantic_type": "Tipo inferido",
                "non_null": "No nulos",
                "missing": "Faltantes",
                "missing_pct": "Faltantes (%)",
                "unique": "Únicos",
            }
        )
        with st.expander("Tipos de columnas inferidos", expanded=False):
            st.dataframe(type_df, width="stretch", hide_index=True)
    if inconsistencies:
        with st.expander("Posibles inconsistencias de tipo", expanded=True):
            for item in inconsistencies:
                st.write(f"• {item}")


def _render_kpis(kpis: dict[str, Any]) -> None:
    rows = [
        ("Ingresos totales", _format_currency(_get(kpis, "total_revenue", "revenue"))),
        ("Pedidos", _format_number(_get(kpis, "total_orders", "order_count"))),
        ("Ticket promedio", _format_currency(_get(kpis, "average_order_value", "aov"))),
        ("Unidades", _format_number(_get(kpis, "total_units", "units"))),
        ("Margen bruto", _format_currency(_get(kpis, "gross_margin"))),
        ("Cancelación", _format_percent(_get(kpis, "cancellation_rate", "cancel_rate"))),
        (
            "Procesamiento promedio",
            (
                f"{_format_number(_get(kpis, 'average_processing_time_hours', 'average_processing_time', 'avg_processing_time'), 1)} h"
                if _get(
                    kpis,
                    "average_processing_time_hours",
                    "average_processing_time",
                    "avg_processing_time",
                )
                is not None
                else "No disponible"
            ),
        ),
    ]
    st.markdown("**Volumen y valor**")
    first = st.columns(4)
    for index, (label, value) in enumerate(rows):
        if index < 4:
            first[index].metric(label, value)
    st.markdown("**Eficiencia operativa**")
    second = st.columns(3)
    for index, (label, value) in enumerate(rows[4:]):
        second[index].metric(label, value)


def _analysis_config_controls() -> AnalysisConfig:
    with st.expander("Configurar análisis comparativo", expanded=False):
        st.caption(
            "Estos umbrales afectan solo diagnósticos determinísticos; no modifican los datos de origen."
        )
        c1, c2, c3 = st.columns(3)
        sla_hours = c1.number_input(
            "SLA de procesamiento (h)",
            min_value=1.0,
            max_value=720.0,
            value=24.0,
            step=1.0,
        )
        cancellation_alert = c2.number_input(
            "Umbral de cancelación alta (%)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=1.0,
        )
        trend_deviation = c3.number_input(
            "Desviación mínima de tendencia (%)",
            min_value=1.0,
            max_value=500.0,
            value=50.0,
            step=5.0,
        )
    return AnalysisConfig(
        sla_hours=float(sla_hours),
        cancellation_alert_rate=float(cancellation_alert),
        trend_deviation_pct=float(trend_deviation),
    )


def _format_analysis_value(value: Any, unit: str) -> str:
    if value is None or pd.isna(value):
        return "—"
    if unit == "currency":
        return _format_currency(value)
    if unit == "percent":
        return _format_percent(value)
    if unit == "hours":
        return f"{_format_number(value, 1)} h"
    return _format_number(value, 1 if unit == "count" and not float(value).is_integer() else 0)


def _render_advanced_analysis(analysis: dict[str, Any]) -> None:
    period = analysis.get("period", {})
    period_metrics = analysis.get("period_metrics")
    if not period.get("available") or not isinstance(period_metrics, pd.DataFrame) or period_metrics.empty:
        st.info(period.get("reason") or "No hay dos períodos comparables disponibles.")
    else:
        st.markdown("**Comparación período contra período**")
        st.caption(str(period.get("note", "")))
        st.caption(
            f"Actual: {period.get('current_label')} · Anterior: {period.get('previous_label')}"
        )
        rows: list[dict[str, Any]] = []
        for _, row in period_metrics.iterrows():
            unit = str(row["unit"])
            absolute = row["absolute_change"]
            if pd.isna(absolute):
                absolute_text = "—"
            elif unit == "percent":
                absolute_text = f"{float(absolute):+.1f} pp".replace(".", ",")
            else:
                absolute_text = _format_analysis_value(float(absolute), unit)
                if float(absolute) > 0:
                    absolute_text = "+" + absolute_text
            relative = row["relative_change_pct"]
            relative_text = (
                f"{float(relative):+.1f}%".replace(".", ",")
                if not pd.isna(relative)
                else "—"
            )
            rows.append(
                {
                    "Indicador": row["label"],
                    "Actual": _format_analysis_value(row["current"], unit),
                    "Anterior": _format_analysis_value(row["previous"], unit),
                    "Cambio absoluto": absolute_text,
                    "Cambio relativo": relative_text,
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    sla = analysis.get("sla", {})
    st.markdown("**Cumplimiento de SLA de procesamiento**")
    if isinstance(sla, dict) and sla.get("available"):
        sla_cols = st.columns(4)
        sla_cols[0].metric("SLA configurado", f"{_format_number(sla['threshold_hours'], 1)} h")
        sla_cols[1].metric("Incumplimientos", _format_number(sla["breach_count"]))
        sla_cols[2].metric("Tasa de incumplimiento", _format_percent(sla["breach_rate"]))
        sla_cols[3].metric("P95 procesamiento", f"{_format_number(sla['p95_hours'], 1)} h")
    else:
        st.info((sla or {}).get("reason", "No hay tiempos de procesamiento para evaluar SLA."))

    segment_summary = analysis.get("segment_summary")
    if isinstance(segment_summary, pd.DataFrame) and not segment_summary.empty:
        st.markdown("**Diagnóstico por segmento**")
        labels = list(dict.fromkeys(segment_summary["segment_label"].astype(str).tolist()))
        selected_label = st.selectbox("Dimensión de segmento", labels, key="phase4_segment_dimension")
        selected = segment_summary.loc[segment_summary["segment_label"].eq(selected_label)].copy()
        display = selected[
            [
                "segment",
                "orders",
                "revenue",
                "gross_margin",
                "gross_margin_rate",
                "cancellation_rate",
                "average_processing_time_hours",
                "sla_breach_rate",
                "eligible_for_highlight",
            ]
        ].rename(
            columns={
                "segment": selected_label,
                "orders": "Pedidos",
                "revenue": "Ingresos",
                "gross_margin": "Margen bruto",
                "gross_margin_rate": "Margen (%)",
                "cancellation_rate": "Cancelación (%)",
                "average_processing_time_hours": "Proceso prom. (h)",
                "sla_breach_rate": "Incumple SLA (%)",
                "eligible_for_highlight": "Base suficiente",
            }
        )
        st.dataframe(display, width="stretch", hide_index=True)

        highlightable = selected.loc[selected["eligible_for_highlight"].fillna(False)]
        reference_bits: list[str] = []
        valid_cancel = highlightable.dropna(subset=["cancellation_rate"])
        if not valid_cancel.empty:
            row = valid_cancel.loc[valid_cancel["cancellation_rate"].idxmax()]
            reference_bits.append(
                f"mayor cancelación: {row['segment']} ({_format_percent(row['cancellation_rate'])})"
            )
        valid_processing = highlightable.dropna(subset=["average_processing_time_hours"])
        if not valid_processing.empty:
            row = valid_processing.loc[valid_processing["average_processing_time_hours"].idxmax()]
            reference_bits.append(
                f"mayor tiempo medio: {row['segment']} "
                f"({_format_number(row['average_processing_time_hours'], 1)} h)"
            )
        valid_margin = highlightable.dropna(subset=["gross_margin"])
        if not valid_margin.empty:
            row = valid_margin.loc[valid_margin["gross_margin"].idxmax()]
            reference_bits.append(
                f"mayor margen bruto: {row['segment']} ({_format_currency(row['gross_margin'])})"
            )
        if reference_bits:
            st.caption("Referencias del segmento · " + " · ".join(reference_bits))

        threshold = analysis.get("config", {}).get("cancellation_alert_rate")
        flagged_cancel = highlightable.loc[highlightable["high_cancellation"].fillna(False)]
        if threshold is not None and not flagged_cancel.empty:
            names = ", ".join(flagged_cancel["segment"].astype(str).tolist())
            st.warning(
                f"Cancelación igual o superior al umbral configurado ({_format_percent(threshold)}): {names}."
            )

        comparison = analysis.get("segment_comparison")
        if isinstance(comparison, pd.DataFrame) and not comparison.empty:
            changes = comparison.loc[comparison["segment_label"].eq(selected_label)].copy()
            if not changes.empty:
                st.caption("Cambios del período actual frente al período anterior comparable.")
                st.dataframe(
                    changes[
                        [
                            "segment",
                            "eligible_for_highlight",
                            "revenue_change_pct",
                            "margin_change_pct",
                            "cancellation_change_pp",
                            "processing_change_pct",
                            "sla_breach_change_pp",
                        ]
                    ].rename(
                        columns={
                            "segment": selected_label,
                            "eligible_for_highlight": "Base suficiente",
                            "revenue_change_pct": "Ingresos Δ (%)",
                            "margin_change_pct": "Margen Δ (%)",
                            "cancellation_change_pp": "Cancelación Δ (pp)",
                            "processing_change_pct": "Proceso Δ (%)",
                            "sla_breach_change_pp": "Incumple SLA Δ (pp)",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )
                eligible_changes = changes.loc[changes["eligible_for_highlight"].fillna(False)]
                deterioration: list[str] = []
                cancel_up = eligible_changes.dropna(subset=["cancellation_change_pp"])
                cancel_up = cancel_up.loc[cancel_up["cancellation_change_pp"].gt(0)]
                if not cancel_up.empty:
                    row = cancel_up.loc[cancel_up["cancellation_change_pp"].idxmax()]
                    deterioration.append(
                        f"mayor aumento de cancelación: {row['segment']} "
                        f"({float(row['cancellation_change_pp']):+.1f} pp)"
                    )
                processing_up = eligible_changes.dropna(subset=["processing_change_pct"])
                processing_up = processing_up.loc[processing_up["processing_change_pct"].gt(0)]
                if not processing_up.empty:
                    row = processing_up.loc[processing_up["processing_change_pct"].idxmax()]
                    deterioration.append(
                        f"mayor aumento de procesamiento: {row['segment']} "
                        f"({float(row['processing_change_pct']):+.1f}%)"
                    )
                revenue_down = eligible_changes.dropna(subset=["revenue_change_pct"])
                revenue_down = revenue_down.loc[revenue_down["revenue_change_pct"].lt(0)]
                if not revenue_down.empty:
                    row = revenue_down.loc[revenue_down["revenue_change_pct"].idxmin()]
                    deterioration.append(
                        f"mayor caída de ingresos: {row['segment']} "
                        f"({float(row['revenue_change_pct']):+.1f}%)"
                    )
                if deterioration:
                    st.caption("Señales de deterioro · " + " · ".join(deterioration))


def _render_trend_anomalies(analysis: dict[str, Any]) -> None:
    trends = analysis.get("trend_anomalies")
    if not isinstance(trends, pd.DataFrame) or trends.empty:
        st.success("No se detectaron desviaciones relevantes contra la línea base histórica reciente.")
        return
    st.warning(
        f"Se detectaron {len(trends)} días con ingresos alejados de su línea base histórica reciente."
    )
    display = trends.copy()
    display["date"] = pd.to_datetime(display["date"]).dt.strftime("%d-%m-%Y")
    display = display.rename(
        columns={
            "date": "Fecha",
            "value": "Ingresos",
            "baseline": "Línea base",
            "deviation_pct": "Desviación (%)",
            "direction": "Dirección",
            "method": "Método",
        }
    )
    st.dataframe(
        display[["Fecha", "Ingresos", "Línea base", "Desviación (%)", "Dirección", "Método"]],
        width="stretch",
        hide_index=True,
    )


def _render_anomalies(findings: list[Any]) -> None:
    if not findings:
        st.success("No se detectaron anomalías relevantes con las reglas actuales.")
        return
    st.warning(
        f"Se detectaron {len(findings)} observaciones fuera del patrón habitual. "
        "Revísalas como señales para investigar, no como causas confirmadas."
    )
    rows: list[dict[str, Any]] = []
    for finding in findings[:20]:
        if isinstance(finding, dict):
            rows.append(
                {
                    "Hallazgo": (
                        finding.get("message")
                        or finding.get("finding")
                        or finding.get("description")
                        or str(finding)
                    ),
                    "Campo": finding.get("column"),
                    "Valor": finding.get("value"),
                    "Referencia": finding.get("median"),
                    "Método": finding.get("method"),
                }
            )
        else:
            rows.append({"Hallazgo": str(finding)})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if len(findings) > 20:
        st.caption(f"Se muestran 20 de {len(findings)} observaciones detectadas.")


def _load_selected_data() -> tuple[pd.DataFrame | None, str | None]:
    action_col, upload_col = st.columns(2, gap="large")
    with action_col:
        st.markdown("**Explorar con la muestra incluida**")
        st.caption("Carga 1.500 operaciones sintéticas con problemas conocidos.")
        if st.button(
            "Probar con datos de ejemplo",
            type="primary",
            width="stretch",
        ):
            st.session_state["use_sample"] = True
            st.session_state["uploaded_name"] = None
    with upload_col:
        st.markdown("**Analizar tu propia planilla**")
        uploaded = st.file_uploader(
            "Subir archivo CSV o XLSX",
            type=["csv", "xlsx"],
            help="Formatos aceptados: CSV y XLSX.",
        )
        if uploaded is not None:
            st.session_state["use_sample"] = False
            st.session_state["uploaded_name"] = uploaded.name

    try:
        if uploaded is not None:
            return load_data(uploaded, filename=uploaded.name), uploaded.name
        if st.session_state.get("use_sample"):
            return load_data(SAMPLE_PATH, filename=SAMPLE_PATH.name), SAMPLE_PATH.name
    except Exception as exc:
        st.error(
            "No pudimos leer el archivo. Verifica que no esté vacío, dañado o en un formato "
            "distinto de CSV/XLSX."
        )
        with st.expander("Detalle técnico", expanded=False):
            st.code(str(exc))
    return None, None


def main() -> None:
    _inject_css()
    st.markdown('<div class="ops-eyebrow">OpsReport · análisis operacional</div>', unsafe_allow_html=True)
    st.title("De una planilla desordenada a un informe operativo en segundos.")
    st.markdown(
        '<div class="ops-lead">'
        "Carga un CSV o Excel. OpsReport revisa la calidad de los datos, calcula indicadores, "
        "detecta señales anómalas y construye un resumen ejecutivo reproducible."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="ops-meta">'
        "<span>CSV y XLSX</span>"
        "<span>Análisis determinístico</span>"
        "<span>Sin API externa</span>"
        "<span>Exportación Excel</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    df, filename = _load_selected_data()
    if df is None:
        st.info(
            "Puedes comenzar con la muestra incluida o cargar tu propia planilla. "
            "El análisis se ejecuta localmente en esta sesión."
        )
        return

    if df.empty:
        st.error("El archivo no contiene filas para analizar.")
        return

    st.markdown(
        f'<div class="ops-file">▦ Analizando: <strong>{escape(filename or "")}</strong></div>',
        unsafe_allow_html=True,
    )
    column_map, _ = _render_schema_mapping(df, filename)
    st.divider()
    analysis_config = _analysis_config_controls()

    try:
        profile = profile_dataframe(df)
        quality = assess_data_quality(df, column_map=column_map)
        kpis = calculate_kpis(df, column_map=column_map)
        findings = detect_anomalies(df, column_map=column_map)
        advanced_analysis = build_operational_analysis(
            df,
            column_map=column_map,
            config=analysis_config,
        )
        summary = generate_management_narrative(
            df,
            profile=profile,
            metrics=kpis,
            anomalies=findings,
            column_map=column_map,
        )
    except Exception as exc:
        st.error("El archivo se pudo leer, pero ocurrió un problema durante el análisis.")
        with st.expander("Detalle técnico", expanded=False):
            st.code(str(exc))
        return

    st.header("1. Vista general")
    st.caption("Cobertura del archivo y una muestra rápida de los registros recibidos.")
    date_range = _date_range(df, column_map)
    overview = st.columns(3)
    overview[0].metric("Filas", _format_number(len(df)))
    overview[1].metric("Columnas", _format_number(len(df.columns)))
    overview[2].metric(
        "Período",
        f"{date_range[0]} → {date_range[1]}" if date_range else "No detectado",
    )
    with st.expander("Ver vista previa de los datos", expanded=False):
        st.dataframe(df.head(20), width="stretch", hide_index=True)

    st.header("2. Calidad de datos")
    st.caption("Diagnóstico reproducible de completitud, duplicados y consistencia básica.")
    _render_quality(quality, profile)

    st.header("3. Indicadores clave")
    st.caption("Métricas disponibles según las columnas que OpsReport pudo interpretar.")
    _render_kpis(kpis)
    unavailable = [
        key
        for key, value in kpis.items()
        if value is None and key not in {"warnings", "available_metrics"}
    ]
    if unavailable:
        st.caption(
            "Algunos indicadores no están disponibles porque el archivo no contiene "
            "las columnas necesarias o no tiene valores interpretables."
        )

    st.header("4. Comparación y diagnóstico por segmento")
    st.caption(
        "Cambios frente al período anterior, desempeño por segmento y cumplimiento del SLA configurado."
    )
    _render_advanced_analysis(advanced_analysis)

    st.header("5. Visualizaciones")
    st.caption("Vistas interactivas construidas directamente desde el archivo analizado.")
    charts = _charts(df, column_map)
    if not charts:
        st.info(
            "No encontramos suficientes columnas compatibles para construir visualizaciones "
            "operativas con este archivo."
        )
    else:
        wide_charts = [
            (title, fig)
            for title, fig in charts
            if title == "Ingresos a lo largo del tiempo"
        ]
        compact_charts = [
            (title, fig)
            for title, fig in charts
            if title != "Ingresos a lo largo del tiempo"
        ]
        for title, fig in wide_charts:
            st.subheader(title)
            st.plotly_chart(fig, width="stretch")
        for index in range(0, len(compact_charts), 2):
            cols = st.columns(2)
            for offset, (title, fig) in enumerate(compact_charts[index : index + 2]):
                with cols[offset]:
                    st.subheader(title)
                    st.plotly_chart(fig, width="stretch")

    st.header("6. Anomalías y señales para investigar")
    st.caption("Outliers por registro y desviaciones contra una línea base histórica reproducible.")
    st.markdown("**Observaciones atípicas por registro**")
    _render_anomalies(findings)
    st.markdown("**Desviaciones de tendencia diaria**")
    _render_trend_anomalies(advanced_analysis)

    st.header("7. Resumen ejecutivo")
    st.caption("Narrativa determinística construida a partir de métricas y hallazgos calculados.")
    if isinstance(summary, (list, tuple)):
        summary_text = " ".join(str(item) for item in summary)
    else:
        summary_text = str(summary)
    safe_summary = escape(summary_text).replace("\n\n", "<br><br>").replace("\n", "<br>")
    st.markdown(f'<div class="ops-summary">{safe_summary}</div>', unsafe_allow_html=True)

    st.header("8. Exportar análisis")
    st.caption("Descarga un libro Excel con contexto, métricas, diagnósticos y datos analizados.")
    try:
        workbook = build_excel_export(
            df,
            profile=profile,
            metrics=kpis,
            anomalies=findings,
            context={
                "source_file": filename,
                "sla_hours": analysis_config.sla_hours,
                "cancellation_alert_rate": analysis_config.cancellation_alert_rate,
                "trend_deviation_pct": analysis_config.trend_deviation_pct,
                "column_mapping": {
                    FIELD_LABELS[field]: column
                    for field, column in column_map.items()
                    if column
                },
            },
            column_map=column_map,
            analysis=advanced_analysis,
        )
        st.download_button(
            "Descargar informe Excel",
            data=workbook,
            file_name=f"OpsReport_{Path(filename or 'analisis').stem}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )
        st.markdown(
            '<div class="ops-note">El archivo incluye contexto, indicadores, comparaciones, '
            "diagnósticos por segmento, SLA, tendencias y los datos analizados.</div>",
            unsafe_allow_html=True,
        )
    except Exception as exc:
        st.warning("El análisis está disponible, pero no pudimos generar el Excel de exportación.")
        with st.expander("Detalle técnico", expanded=False):
            st.code(str(exc))


if __name__ == "__main__":
    main()
