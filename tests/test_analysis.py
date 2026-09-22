import pandas as pd
import pytest

from src.analysis import AnalysisConfig, build_operational_analysis


def _comparison_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["A1", "A2", "A3", "S1", "S2", "S3"],
            "date": [
                "2026-08-01",
                "2026-08-05",
                "2026-08-10",
                "2026-09-01",
                "2026-09-05",
                "2026-09-10",
            ],
            "revenue": [100, 100, 100, 100, 200, 300],
            "cost": [60, 60, 60, 50, 100, 150],
            "units": [1, 1, 1, 1, 2, 3],
            "status": ["Completed", "Cancelled", "Completed", "Completed", "Completed", "Cancelled"],
            "region": ["North", "South", "North", "North", "South", "North"],
            "category": ["A", "B", "A", "A", "B", "A"],
            "channel": ["Online", "Store", "Online", "Online", "Store", "Online"],
            "processing_hours": [10, 24, 30, 12, 25, 48],
        }
    )


def test_incomplete_month_compares_same_elapsed_days() -> None:
    analysis = build_operational_analysis(
        _comparison_frame(),
        config=AnalysisConfig(min_segment_size=1),
    )

    period = analysis["period"]
    assert period["available"] is True
    assert period["current_incomplete"] is True
    assert period["current_start"] == pd.Timestamp("2026-09-01")
    assert period["current_end"] == pd.Timestamp("2026-09-10")
    assert period["previous_start"] == pd.Timestamp("2026-08-01")
    assert period["previous_end"] == pd.Timestamp("2026-08-10")

    metrics = analysis["period_metrics"].set_index("metric")
    assert metrics.loc["revenue", "current"] == 600.0
    assert metrics.loc["revenue", "previous"] == 300.0
    assert metrics.loc["revenue", "absolute_change"] == 300.0
    assert metrics.loc["revenue", "relative_change_pct"] == pytest.approx(100.0)
    assert metrics.loc["order_count", "relative_change_pct"] == pytest.approx(0.0)


def test_period_comparison_handles_zero_previous_without_infinite_change() -> None:
    frame = _comparison_frame()
    frame.loc[frame["date"].str.startswith("2026-08"), "revenue"] = 0

    analysis = build_operational_analysis(frame, config=AnalysisConfig(min_segment_size=1))
    revenue = analysis["period_metrics"].set_index("metric").loc["revenue"]

    assert revenue["absolute_change"] == 600.0
    assert pd.isna(revenue["relative_change_pct"])


def test_segment_summary_and_changes_cover_all_supported_dimensions() -> None:
    analysis = build_operational_analysis(
        _comparison_frame(),
        config=AnalysisConfig(sla_hours=24, min_segment_size=1),
    )

    summary = analysis["segment_summary"]
    assert set(summary["segment_type"]) == {"region", "category", "channel", "status"}
    assert summary.loc[summary["segment_type"].eq("status"), "cancellation_rate"].isna().all()

    north = summary.loc[
        summary["segment_type"].eq("region") & summary["segment"].eq("North")
    ].iloc[0]
    assert north["revenue"] == 400.0
    assert north["gross_margin"] == 200.0
    assert north["sla_breach_rate"] == pytest.approx(50.0)

    changes = analysis["segment_comparison"]
    north_change = changes.loc[
        changes["segment_type"].eq("region") & changes["segment"].eq("North")
    ].iloc[0]
    assert north_change["revenue_change_pct"] == pytest.approx(100.0)
    assert north_change["processing_change_pct"] is not None


def test_small_segments_remain_visible_but_are_not_highlight_eligible() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2026-09-01", "2026-09-01"],
            "region": ["North", "South"],
            "revenue": [100, 200],
        }
    )

    analysis = build_operational_analysis(frame, config=AnalysisConfig(min_segment_size=3))
    regions = analysis["segment_summary"].loc[
        analysis["segment_summary"]["segment_type"].eq("region")
    ]

    assert set(regions["segment"]) == {"North", "South"}
    assert not regions["eligible_for_highlight"].any()


def test_sla_uses_strict_greater_than_threshold_and_ignores_invalid_values() -> None:
    frame = pd.DataFrame(
        {
            "processing_hours": [10, 24, 25, 48, None, -1, "bad"],
        }
    )

    analysis = build_operational_analysis(frame, config=AnalysisConfig(sla_hours=24))
    sla = analysis["sla"]

    assert sla["available"] is True
    assert sla["observations"] == 4
    assert sla["breach_count"] == 2
    assert sla["breach_rate"] == pytest.approx(50.0)


def test_time_series_anomaly_uses_prior_history_without_lookahead() -> None:
    dates = pd.date_range("2026-01-01", periods=40, freq="D")
    revenue = [100.0] * 40
    revenue[34] = 1000.0
    frame = pd.DataFrame({"date": dates, "revenue": revenue})
    config = AnalysisConfig(
        trend_window_days=10,
        trend_min_history=5,
        trend_deviation_pct=50,
    )

    analysis = build_operational_analysis(frame, config=config)
    trends = analysis["trend_anomalies"]

    assert list(trends["date"]) == [pd.Timestamp("2026-02-04")]
    assert trends.iloc[0]["baseline"] == pytest.approx(100.0)
    assert trends.iloc[0]["deviation_pct"] == pytest.approx(900.0)

    extended = pd.concat(
        [
            frame,
            pd.DataFrame(
                {
                    "date": pd.date_range("2026-02-10", periods=3, freq="D"),
                    "revenue": [5000.0, 5000.0, 5000.0],
                }
            ),
        ],
        ignore_index=True,
    )
    extended_trends = build_operational_analysis(extended, config=config)["trend_anomalies"]
    earlier = extended_trends.loc[extended_trends["date"].le(pd.Timestamp("2026-02-09"))]
    assert list(earlier["date"]) == [pd.Timestamp("2026-02-04")]


def test_missing_previous_period_degrades_cleanly() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2026-09-02", "2026-09-02"],
            "revenue": [100, 200],
            "region": ["North", "South"],
        }
    )

    analysis = build_operational_analysis(frame)

    assert analysis["period"]["available"] is False
    assert analysis["period_metrics"].empty
    assert analysis["segment_comparison"].empty


@pytest.mark.parametrize(
    "config",
    [
        AnalysisConfig(sla_hours=0),
        AnalysisConfig(cancellation_alert_rate=101),
        AnalysisConfig(trend_deviation_pct=0),
        AnalysisConfig(trend_window_days=1),
        AnalysisConfig(trend_window_days=5, trend_min_history=6),
        AnalysisConfig(min_segment_size=0),
        AnalysisConfig(sla_hours=float("nan")),
    ],
)
def test_invalid_analysis_config_is_rejected(config: AnalysisConfig) -> None:
    with pytest.raises(ValueError):
        build_operational_analysis(pd.DataFrame({"date": ["2026-09-01"]}), config=config)
