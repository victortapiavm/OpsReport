from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.anomalies import detect_anomalies
from src.exports import build_excel_export
from src.ingestion import load_data
from src.metrics import calculate_kpis
from src.narrative import generate_management_narrative
from src.profiling import profile_dataframe
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


def _find_column(df: pd.DataFrame, *candidates: str) -> str | None:
    normalized = {str(column).strip().lower(): str(column) for column in df.columns}
    for candidate in candidates:
        match = normalized.get(candidate.lower())
        if match is not None:
            return match
    return None


def _date_range(df: pd.DataFrame) -> tuple[str, str] | None:
    date_column = _find_column(df, "date", "fecha", "order_date", "fecha_pedido")
    if not date_column:
        return None
    parsed = pd.to_datetime(df[date_column], errors="coerce")
    parsed = parsed.dropna()
    if parsed.empty:
        return None
    return parsed.min().strftime("%d-%m-%Y"), parsed.max().strftime("%d-%m-%Y")


def _charts(df: pd.DataFrame) -> list[tuple[str, Any]]:
    charts: list[tuple[str, Any]] = []
    date_col = _find_column(df, "date", "fecha")
    revenue_col = _find_column(df, "revenue", "revenue_clp", "ingresos", "ventas")
    order_col = _find_column(df, "order_id", "pedido_id", "id_pedido")
    region_col = _find_column(df, "region", "región")
    status_col = _find_column(df, "status", "estado")
    category_col = _find_column(df, "category", "categoria", "categoría")
    processing_col = _find_column(
        df,
        "processing_hours",
        "processing_time_hours",
        "horas_procesamiento",
        "processing_time",
    )

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
        tmp["_cancelled"] = (
            tmp[status_col]
            .astype("string")
            .str.strip()
            .str.lower()
            .isin({"cancelled", "canceled", "cancelado", "cancelada"})
        )
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
        if isinstance(issue, dict) and issue.get("code") == "invalid_numeric_values"
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

    try:
        profile = profile_dataframe(df)
        quality = assess_data_quality(df)
        kpis = calculate_kpis(df)
        findings = detect_anomalies(df)
        summary = generate_management_narrative(
            df,
            profile=profile,
            metrics=kpis,
            anomalies=findings,
        )
    except Exception as exc:
        st.error("El archivo se pudo leer, pero ocurrió un problema durante el análisis.")
        with st.expander("Detalle técnico", expanded=False):
            st.code(str(exc))
        return

    st.markdown(
        f'<div class="ops-file">▦ Analizando: <strong>{escape(filename or "")}</strong></div>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.header("1. Vista general")
    st.caption("Cobertura del archivo y una muestra rápida de los registros recibidos.")
    date_range = _date_range(df)
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

    st.header("4. Visualizaciones")
    st.caption("Vistas interactivas construidas directamente desde el archivo analizado.")
    charts = _charts(df)
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

    st.header("5. Anomalías y señales para investigar")
    st.caption("Outliers estadísticos simples y reproducibles sobre variables operativas.")
    _render_anomalies(findings)

    st.header("6. Resumen ejecutivo")
    st.caption("Narrativa determinística construida a partir de métricas y hallazgos calculados.")
    if isinstance(summary, (list, tuple)):
        summary_text = " ".join(str(item) for item in summary)
    else:
        summary_text = str(summary)
    safe_summary = escape(summary_text).replace("\n\n", "<br><br>").replace("\n", "<br>")
    st.markdown(f'<div class="ops-summary">{safe_summary}</div>', unsafe_allow_html=True)

    st.header("7. Exportar análisis")
    st.caption("Descarga un libro Excel con contexto, métricas, diagnósticos y datos analizados.")
    try:
        workbook = build_excel_export(
            df,
            profile=profile,
            metrics=kpis,
            anomalies=findings,
            context={"source_file": filename},
        )
        st.download_button(
            "Descargar informe Excel",
            data=workbook,
            file_name=f"OpsReport_{Path(filename or 'analisis').stem}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )
        st.markdown(
            '<div class="ops-note">El archivo incluye contexto, indicadores, diagnósticos y '
            "los datos analizados.</div>",
            unsafe_allow_html=True,
        )
    except Exception as exc:
        st.warning("El análisis está disponible, pero no pudimos generar el Excel de exportación.")
        with st.expander("Detalle técnico", expanded=False):
            st.code(str(exc))


if __name__ == "__main__":
    main()
