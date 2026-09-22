"""Explainable semantic column recognition for operational spreadsheets.

The recognizer intentionally uses transparent header aliases plus lightweight
type checks. It does not guess business meaning from values alone, so uncertain
files can be resolved through the Streamlit mapping UI instead of silently
assigning a plausible-looking numeric or categorical column.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping
import re
import unicodedata

import pandas as pd


SEMANTIC_FIELDS = (
    "order_id",
    "date",
    "revenue",
    "cost",
    "quantity",
    "status",
    "region",
    "channel",
    "category",
    "processing_time_hours",
)

FIELD_LABELS: dict[str, str] = {
    "order_id": "ID de pedido",
    "date": "Fecha",
    "revenue": "Ingresos",
    "cost": "Costo",
    "quantity": "Unidades",
    "status": "Estado",
    "region": "Región",
    "channel": "Canal",
    "category": "Categoría",
    "processing_time_hours": "Tiempo de procesamiento (h)",
}

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "order_id": (
        "order_id",
        "order id",
        "id_order",
        "id pedido",
        "id_pedido",
        "pedido_id",
        "pedido",
        "orden_id",
        "id orden",
        "nro pedido",
        "numero pedido",
        "n pedido",
        "order_number",
        "order number",
        "operation_id",
        "id operacion",
    ),
    "date": (
        "date",
        "order_date",
        "order date",
        "fecha",
        "fecha_pedido",
        "fecha pedido",
        "fecha_operacion",
        "fecha operacion",
        "operation_date",
        "transaction_date",
        "transaction date",
    ),
    "revenue": (
        "revenue",
        "revenue_clp",
        "sales",
        "venta",
        "ventas",
        "ingreso",
        "ingresos",
        "ingresos_clp",
        "facturacion",
        "facturación",
        "monto_venta",
        "monto venta",
        "total_venta",
        "total venta",
        "sales_amount",
        "net_sales",
        "net sales",
        "total_revenue",
    ),
    "quantity": (
        "quantity",
        "qty",
        "units",
        "unit_count",
        "cantidad",
        "unidades",
        "cantidad_unidades",
        "units_sold",
        "units sold",
    ),
    "cost": (
        "cost",
        "costs",
        "costo",
        "costos",
        "total_cost",
        "costo_total",
        "costo total",
        "cost_amount",
        "cogs",
    ),
    "status": (
        "status",
        "order_status",
        "estado",
        "estado_pedido",
        "estado pedido",
        "situacion",
        "order_state",
    ),
    "region": (
        "region",
        "región",
        "zona",
        "territorio",
        "territory",
        "market_region",
        "market region",
        "area",
        "área",
    ),
    "channel": (
        "channel",
        "sales_channel",
        "sales channel",
        "canal",
        "canal_venta",
        "canal venta",
        "origen",
        "source_channel",
    ),
    "category": (
        "category",
        "categoria",
        "categoría",
        "product_category",
        "categoria_producto",
        "tipo_producto",
        "product_family",
        "product family",
        "familia_producto",
        "familia producto",
    ),
    "processing_time_hours": (
        "processing_time_hours",
        "processing_hours",
        "processing hours",
        "processing_time",
        "lead_time_hours",
        "lead time hours",
        "tiempo_procesamiento_horas",
        "tiempo procesamiento horas",
        "tiempo_procesamiento",
        "tiempo_proceso_horas",
        "tiempo_proceso",
        "horas_procesamiento",
        "horas_proceso",
    ),
    "created_at": (
        "created_at",
        "created",
        "order_date",
        "fecha_creacion",
        "fecha creación",
        "fecha_pedido",
        "fecha pedido",
        "fecha_inicio",
    ),
    "completed_at": (
        "completed_at",
        "completed",
        "completion_date",
        "fecha_completado",
        "fecha completado",
        "fecha_cierre",
        "fecha cierre",
        "fecha_fin",
    ),
}

NUMERIC_FIELDS = {"revenue", "cost", "quantity", "processing_time_hours"}
DATE_FIELDS = {"date", "created_at", "completed_at"}
CATEGORICAL_FIELDS = {"status", "region", "channel", "category"}
CANCELLED_STATUSES = {
    "cancelado",
    "cancelada",
    "cancelled",
    "canceled",
    "anulado",
    "anulada",
}


@dataclass(frozen=True)
class ColumnInference:
    """One explainable semantic-role proposal for a dataframe column."""

    field: str
    column: str | None
    confidence: Literal["high", "medium", "low", "none"]
    score: float
    reason: str
    alternatives: tuple[str, ...] = ()
    requires_confirmation: bool = False


def normalize_name(value: Any) -> str:
    """Normalize a header for accent/case/separator-insensitive comparison."""

    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def resolve_column(
    df: pd.DataFrame,
    canonical_name: str,
    column_map: Mapping[str, str | None] | None = None,
) -> str | None:
    """Resolve a semantic field to an actual dataframe column.

    When ``column_map`` explicitly contains a field, that choice is authoritative.
    An empty or invalid mapped value therefore means "do not map this field" and
    prevents fallback alias guessing. Without an explicit choice, common
    Spanish/English aliases are matched deterministically.
    """

    if column_map is not None and canonical_name in column_map:
        mapped = column_map[canonical_name]
        return mapped if mapped in df.columns else None

    normalized_columns: dict[str, list[str]] = {}
    for column in df.columns:
        normalized_columns.setdefault(normalize_name(column), []).append(str(column))
    candidates = COLUMN_ALIASES.get(canonical_name, (canonical_name,))
    for candidate in candidates:
        actual = normalized_columns.get(normalize_name(candidate), [])
        if len(actual) == 1:
            return actual[0]
        if len(actual) > 1:
            return None
    return None


def cancelled_status_mask(series: pd.Series) -> pd.Series:
    """Return the shared deterministic cancellation predicate."""

    normalized = series.astype("string").str.strip().map(normalize_name)
    return normalized.isin(CANCELLED_STATUSES) | normalized.str.startswith("cancel", na=False)


def infer_schema(df: pd.DataFrame) -> dict[str, ColumnInference]:
    """Propose a unique source column for each supported semantic field.

    Header meaning drives the match. Type compatibility can strengthen or weaken
    a named candidate, but a column is never assigned solely because it contains
    numbers, dates, or low-cardinality values.
    """

    ranked: dict[str, list[tuple[float, str, str]]] = {}
    for field in SEMANTIC_FIELDS:
        candidates: list[tuple[float, str, str]] = []
        for column in df.columns:
            actual = str(column)
            header_score, reason = _header_score(actual, COLUMN_ALIASES[field])
            if header_score <= 0:
                continue
            score = max(0.0, min(1.0, header_score + _type_adjustment(df[actual], field)))
            candidates.append((round(score, 3), actual, reason))
        ranked[field] = sorted(candidates, key=lambda item: (-item[0], item[1].lower()))

    assignments: dict[str, tuple[float, str, str]] = {}
    claimed_columns: set[str] = set()
    pairs = sorted(
        (
            (score, field_index, field, column, reason)
            for field_index, field in enumerate(SEMANTIC_FIELDS)
            for score, column, reason in ranked[field]
            if score >= 0.45
        ),
        key=lambda item: (-item[0], item[1], item[3].lower()),
    )
    for score, _, field, column, reason in pairs:
        if field in assignments or column in claimed_columns:
            continue
        assignments[field] = (score, column, reason)
        claimed_columns.add(column)

    result: dict[str, ColumnInference] = {}
    for field in SEMANTIC_FIELDS:
        assignment = assignments.get(field)
        if assignment is None:
            result[field] = ColumnInference(
                field=field,
                column=None,
                confidence="none",
                score=0.0,
                reason="No se encontró una coincidencia explicable por nombre.",
            )
            continue

        score, column, reason = assignment
        alternatives = tuple(item[1] for item in ranked[field] if item[1] != column)[:3]
        competing_scores = [item[0] for item in ranked[field] if item[1] != column]
        margin = score - competing_scores[0] if competing_scores else score
        confidence = _confidence(score, margin)
        result[field] = ColumnInference(
            field=field,
            column=column,
            confidence=confidence,
            score=score,
            reason=reason,
            alternatives=alternatives,
            requires_confirmation=confidence != "high",
        )
    return result


def suggested_mapping(
    inferences: Mapping[str, ColumnInference],
    include_medium: bool = False,
) -> dict[str, str]:
    """Build the safe automatic mapping from inference results."""

    accepted = {"high", "medium"} if include_medium else {"high"}
    return {
        field: inference.column
        for field, inference in inferences.items()
        if inference.column is not None and inference.confidence in accepted
    }


def validate_column_mapping(
    df: pd.DataFrame,
    mapping: Mapping[str, str | None],
) -> list[str]:
    """Return human-readable errors for an interactive mapping selection."""

    errors: list[str] = []
    selected_columns: dict[str, str] = {}
    for field, column in mapping.items():
        if field not in SEMANTIC_FIELDS:
            errors.append(f"Rol semántico desconocido: {field}.")
            continue
        if not column:
            continue
        if column not in df.columns:
            errors.append(f"La columna «{column}» ya no existe en el archivo.")
            continue
        previous = selected_columns.get(column)
        if previous is not None:
            errors.append(
                f"La columna «{column}» está asignada a «{FIELD_LABELS[previous]}» y "
                f"«{FIELD_LABELS[field]}». Cada columna debe tener un solo rol."
            )
        else:
            selected_columns[column] = field
    return errors


def _header_score(column: str, aliases: tuple[str, ...]) -> tuple[float, str]:
    normalized_column = normalize_name(column)
    column_tokens = set(normalized_column.split("_")) if normalized_column else set()

    best_score = 0.0
    best_reason = ""
    for alias in aliases:
        normalized_alias = normalize_name(alias)
        if normalized_column == normalized_alias:
            return 0.96, f"El encabezado coincide con «{alias}»."

        alias_tokens = set(normalized_alias.split("_")) if normalized_alias else set()
        if not alias_tokens or not column_tokens:
            continue
        overlap = len(alias_tokens & column_tokens) / len(alias_tokens | column_tokens)
        if alias_tokens.issubset(column_tokens) and len(alias_tokens) >= 2:
            score = 0.78
        elif overlap >= 0.5:
            score = 0.56 + 0.18 * overlap
        elif len(alias_tokens) == 1:
            token = next(iter(alias_tokens))
            score = 0.5 if len(token) >= 4 and token in column_tokens else 0.0
        else:
            score = 0.0
        if score > best_score:
            best_score = score
            best_reason = f"El encabezado comparte términos con «{alias}»."
    return best_score, best_reason


def _type_adjustment(series: pd.Series, field: str) -> float:
    nonblank = series.dropna()
    if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
        nonblank = nonblank[nonblank.astype("string").str.strip().ne("")]
    if nonblank.empty:
        return 0.0

    if field in NUMERIC_FIELDS:
        ratio = float(pd.to_numeric(nonblank, errors="coerce").notna().mean())
        if ratio >= 0.9:
            return 0.04
        if ratio >= 0.7:
            return 0.01
        if ratio < 0.5:
            return -0.18
        return -0.05

    if field in DATE_FIELDS:
        ratio = float(pd.to_datetime(nonblank, errors="coerce").notna().mean())
        if ratio >= 0.9:
            return 0.04
        if ratio >= 0.7:
            return 0.01
        if ratio < 0.5:
            return -0.18
        return -0.05

    if field == "order_id":
        uniqueness = float(nonblank.nunique(dropna=True) / len(nonblank))
        return 0.02 if uniqueness >= 0.7 else 0.0

    if field in CATEGORICAL_FIELDS:
        unique = int(nonblank.nunique(dropna=True))
        threshold = max(20, int(len(nonblank) * 0.25))
        return 0.02 if unique <= threshold else 0.0
    return 0.0


def _confidence(score: float, margin: float) -> Literal["high", "medium", "low"]:
    if score >= 0.93 and margin >= 0.12:
        return "high"
    if score >= 0.68 and margin >= 0.08:
        return "medium"
    return "low"
