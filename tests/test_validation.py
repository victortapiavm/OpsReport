import pandas as pd
import pytest

from src.validation import assess_data_quality


def test_quality_score_is_100_for_clean_unique_data() -> None:
    df = pd.DataFrame(
        {
            "id_pedido": ["A", "B", "C"],
            "ventas": [100, 200, 300],
            "cantidad": [1, 2, 3],
            "costo": [50, 120, 180],
        }
    )

    report = assess_data_quality(df)

    assert report["score"] == 100.0
    assert report["issues"] == []
    assert report["warnings"] == []
    assert report["missing_count"] == 0
    assert report["missing_percentage"] == 0.0
    assert report["stats"]["missing_required_columns"] == []


def test_quality_report_handles_missing_required_and_nonnumeric_values() -> None:
    df = pd.DataFrame({"ventas": [100, "oops", None], "cantidad": [1, "x", 3]})

    report = assess_data_quality(df)

    assert report["score"] < 100
    assert report["stats"]["missing_required_columns"] == ["order_id"]
    assert report["stats"]["invalid_numeric_values"] == 2
    assert report["missing_count"] == 1
    assert report["missing_percentage"] == pytest.approx(16.7, abs=0.1)
    assert {issue["code"] for issue in report["issues"]} >= {
        "missing_required_column",
        "missing_required_values",
        "invalid_numeric_values",
    }


def test_empty_dataset_scores_zero_without_crashing() -> None:
    report = assess_data_quality(pd.DataFrame(columns=["order_id", "revenue"]))

    assert report["score"] == 0.0
    assert report["issues"][0]["code"] == "empty_dataset"
    assert report["warnings"] == ["El archivo no contiene filas para analizar."]


def test_duplicate_rows_reduce_uniqueness_component() -> None:
    df = pd.DataFrame(
        {
            "order_id": [1, 1],
            "revenue": [10, 10],
            "quantity": [1, 1],
            "cost": [4, 4],
        }
    )

    report = assess_data_quality(df)

    assert report["stats"]["duplicate_rows"] == 1
    assert report["components"]["uniqueness"] == 7.5
    assert report["score"] == 92.5
