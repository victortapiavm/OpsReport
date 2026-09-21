import pandas as pd
import pytest

from src.metrics import calculate_kpis


def test_calculate_kpis_from_spanish_columns() -> None:
    df = pd.DataFrame(
        {
            "ID Pedido": ["A", "A", "B", "C"],
            "Ventas": [100, 50, 200, 150],
            "Cantidad": [1, 2, 3, 1],
            "Costo": [40, 25, 120, 90],
            "Estado": ["Completado", "Completado", "Cancelado", "Completado"],
            "Fecha Pedido": [
                "2026-09-01 08:00",
                "2026-09-01 09:00",
                "2026-09-02 10:00",
                "2026-09-03 12:00",
            ],
            "Fecha Cierre": [
                "2026-09-01 10:00",
                "2026-09-01 13:00",
                "2026-09-02 16:00",
                "2026-09-03 16:00",
            ],
        }
    )

    kpis = calculate_kpis(df)

    assert kpis["row_count"] == 4
    assert kpis["order_count"] == 3
    assert kpis["revenue"] == 500.0
    assert kpis["average_order_value"] == pytest.approx(500 / 3)
    assert kpis["units"] == 7.0
    assert kpis["gross_margin"] == 225.0
    assert kpis["gross_margin_rate"] == 45.0
    assert kpis["cancellation_rate"] == 25.0
    assert kpis["average_processing_time_hours"] == 4.0
    assert kpis["average_processing_time"] == 4.0


def test_zero_revenue_and_zero_orders_do_not_divide_by_zero() -> None:
    df = pd.DataFrame(
        {
            "order_id": [None, ""],
            "revenue": [0, 0],
            "cost": [0, 0],
            "status": [None, ""],
        }
    )

    kpis = calculate_kpis(df)

    assert kpis["order_count"] == 0
    assert kpis["revenue"] == 0.0
    assert kpis["average_order_value"] is None
    assert kpis["gross_margin"] == 0.0
    assert kpis["gross_margin_rate"] is None
    assert kpis["cancellation_rate"] is None


def test_missing_and_nonnumeric_fields_return_none_or_valid_partial_kpis() -> None:
    df = pd.DataFrame(
        {
            "order_id": [1, 2],
            "revenue": ["bad", None],
            "quantity": ["3", "bad"],
        }
    )

    kpis = calculate_kpis(df)

    assert kpis["order_count"] == 2
    assert kpis["revenue"] is None
    assert kpis["average_order_value"] is None
    assert kpis["units"] == 3.0
    assert kpis["gross_margin"] is None
    assert kpis["average_processing_time_hours"] is None


def test_direct_processing_hours_ignore_negative_and_invalid_values() -> None:
    df = pd.DataFrame({"tiempo_procesamiento_horas": [2, 4, -1, "bad"]})

    kpis = calculate_kpis(df)

    assert kpis["average_processing_time_hours"] == 3.0


def test_sample_processing_hours_alias_is_supported() -> None:
    df = pd.DataFrame({"processing_hours": [1.5, 2.5, 4.0]})

    kpis = calculate_kpis(df)

    assert kpis["average_processing_time_hours"] == pytest.approx(8 / 3)
    assert kpis["average_processing_time"] == pytest.approx(8 / 3)
