import pandas as pd

from src.anomalies import detect_anomalies


def test_detects_known_extreme_revenue_outlier() -> None:
    df = pd.DataFrame({"ventas": [100, 102, 98, 101, 99, 1000]})

    findings = detect_anomalies(df)

    assert len(findings) == 1
    assert findings[0]["row_position"] == 5
    assert findings[0]["value"] == 1000.0
    assert findings[0]["direction"] == "high"
    assert findings[0]["severity"] == "warning"
    assert "Valor atípico" in findings[0]["message"]
    assert "ingresos" in findings[0]["message"]


def test_typical_small_variation_is_not_flagged() -> None:
    df = pd.DataFrame({"revenue": [98, 99, 100, 101, 102, 100]})

    assert detect_anomalies(df) == []


def test_constant_baseline_flags_only_deviating_value() -> None:
    df = pd.DataFrame({"quantity": [10, 10, 10, 10, 40]})

    findings = detect_anomalies(df)

    assert len(findings) == 1
    assert findings[0]["value"] == 40.0
    assert findings[0]["method"] == "constant_baseline"


def test_missing_nonnumeric_and_too_small_columns_are_ignored() -> None:
    df = pd.DataFrame(
        {
            "revenue": ["bad", None, 10, 11],
            "description": ["a", "b", "c", "d"],
        }
    )

    assert detect_anomalies(df) == []


def test_explicit_actual_numeric_column_can_be_analyzed() -> None:
    df = pd.DataFrame({"custom_metric": [5, 5, 6, 5, 100]})

    findings = detect_anomalies(df, columns=["custom_metric"])

    assert len(findings) == 1
    assert findings[0]["column"] == "custom_metric"
    assert findings[0]["value"] == 100.0


def test_processing_hours_sample_alias_is_checked_for_outliers() -> None:
    df = pd.DataFrame({"processing_hours": [2, 2, 2, 2, 20]})

    findings = detect_anomalies(df)

    assert len(findings) == 1
    assert findings[0]["canonical_column"] == "processing_time_hours"
    assert findings[0]["column"] == "processing_hours"
