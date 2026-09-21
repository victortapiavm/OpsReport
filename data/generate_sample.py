"""Generate the deterministic OpsReport demo dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260921
BASE_ROWS = 1490
DUPLICATE_ROWS = 10
OUTPUT_PATH = Path(__file__).with_name("sample_operations.csv")


def generate_sample() -> pd.DataFrame:
    """Create 1,500 realistic operational records with intentional data issues."""
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2026-01-01", "2026-08-31", freq="D")
    date_weights = np.linspace(0.75, 1.25, len(dates))
    date_weights *= 1 + 0.18 * np.sin(np.arange(len(dates)) * 2 * np.pi / 30)
    date_weights /= date_weights.sum()

    regions = np.array(["Metropolitana", "Valparaíso", "Biobío", "Maule", "Antofagasta", "La Araucanía"])
    region_probs = np.array([0.43, 0.16, 0.14, 0.10, 0.09, 0.08])
    categories = np.array(["Insumos", "Equipamiento", "Servicios", "Repuestos"])
    category_probs = np.array([0.34, 0.27, 0.25, 0.14])
    channels = np.array(["Web", "Ejecutivo", "Marketplace", "Teléfono"], dtype=object)

    operation_dates = rng.choice(dates.to_numpy(), size=BASE_ROWS, p=date_weights)
    operation_regions = rng.choice(regions, size=BASE_ROWS, p=region_probs)
    operation_categories = rng.choice(categories, size=BASE_ROWS, p=category_probs)
    operation_channels = rng.choice(channels, size=BASE_ROWS, p=[0.42, 0.30, 0.18, 0.10]).astype(object)

    units = np.maximum(1, rng.poisson(lam=4.2, size=BASE_ROWS)).astype(float)
    unit_price = np.select(
        [operation_categories == "Equipamiento", operation_categories == "Servicios", operation_categories == "Repuestos"],
        [175000.0, 92000.0, 128000.0],
        default=54000.0,
    )
    revenue = units * unit_price * rng.lognormal(mean=0, sigma=0.22, size=BASE_ROWS)
    base_cost_ratio = np.select(
        [
            operation_categories == "Equipamiento",
            operation_categories == "Servicios",
            operation_categories == "Repuestos",
        ],
        [0.72, 0.58, 0.82],
        default=0.68,
    )
    base_cost_ratio += np.where(operation_regions == "La Araucanía", 0.04, 0.0)
    base_cost_ratio += np.where(operation_channels == "Marketplace", 0.025, 0.0)
    cost_ratio = np.clip(
        base_cost_ratio * rng.lognormal(mean=0, sigma=0.035, size=BASE_ROWS),
        0.45,
        0.94,
    )
    cost = revenue * cost_ratio

    region_penalty = np.where(operation_regions == "La Araucanía", 1.55, 1.0)
    category_penalty = np.where(operation_categories == "Repuestos", 1.32, 1.0)
    month_number = pd.DatetimeIndex(operation_dates).month.to_numpy()
    seasonal_penalty = np.where(np.isin(month_number, [6, 7]), 1.18, 1.0)
    processing_time = rng.lognormal(mean=np.log(15.5), sigma=0.38, size=BASE_ROWS)
    processing_time *= region_penalty * category_penalty * seasonal_penalty
    sla_hours = np.where(operation_categories == "Servicios", 36.0, 24.0)

    cancellation_probability = np.full(BASE_ROWS, 0.055)
    cancellation_probability += np.where(operation_regions == "La Araucanía", 0.115, 0.0)
    cancellation_probability += np.where(operation_categories == "Repuestos", 0.055, 0.0)
    cancellation_probability += np.where(processing_time > sla_hours * 1.5, 0.06, 0.0)
    cancellation_probability = np.clip(cancellation_probability, 0, 0.40)
    is_cancelled = rng.random(BASE_ROWS) < cancellation_probability
    is_pending = (~is_cancelled) & (rng.random(BASE_ROWS) < 0.045)
    status = np.where(is_cancelled, "Cancelada", np.where(is_pending, "Pendiente", "Completada"))

    dataframe = pd.DataFrame(
        {
            "order_id": [f"OP-{index:05d}" for index in range(1, BASE_ROWS + 1)],
            "date": pd.to_datetime(operation_dates).strftime("%Y-%m-%d"),
            "region": operation_regions,
            "channel": operation_channels,
            "category": operation_categories,
            "status": status,
            "units": units,
            "revenue": revenue.round(0),
            "cost": cost.round(0),
            "processing_hours": processing_time.round(2),
        }
    )

    _inject_missing_values(dataframe, rng)
    _inject_extremes(dataframe, rng)
    _inject_processing_anomalies(dataframe, rng)

    duplicate_source = rng.choice(dataframe.index.to_numpy(), size=DUPLICATE_ROWS, replace=False)
    duplicates = dataframe.loc[duplicate_source].copy()
    dataframe = pd.concat([dataframe, duplicates], ignore_index=True)
    return dataframe


def _inject_missing_values(dataframe: pd.DataFrame, rng: np.random.Generator) -> None:
    for column, fraction in (("channel", 0.025), ("category", 0.018), ("revenue", 0.012), ("cost", 0.009)):
        count = max(1, int(round(len(dataframe) * fraction)))
        indices = rng.choice(dataframe.index.to_numpy(), size=count, replace=False)
        dataframe.loc[indices, column] = np.nan


def _inject_extremes(dataframe: pd.DataFrame, rng: np.random.Generator) -> None:
    extreme_indices = rng.choice(dataframe.index.to_numpy(), size=8, replace=False)
    dataframe.loc[extreme_indices[:4], "units"] *= rng.integers(12, 25, size=4)
    first_multipliers = rng.uniform(6.0, 10.0, size=4)
    dataframe.loc[extreme_indices[:4], "revenue"] *= first_multipliers
    dataframe.loc[extreme_indices[:4], "cost"] *= first_multipliers * rng.uniform(0.96, 1.04, size=4)
    second_multipliers = rng.uniform(8.0, 14.0, size=4)
    dataframe.loc[extreme_indices[4:], "revenue"] *= second_multipliers
    dataframe.loc[extreme_indices[4:], "cost"] *= second_multipliers * rng.uniform(0.96, 1.04, size=4)
    dataframe["units"] = dataframe["units"].round(0)
    dataframe["revenue"] = dataframe["revenue"].round(0)
    dataframe["cost"] = dataframe["cost"].round(0)


def _inject_processing_anomalies(dataframe: pd.DataFrame, rng: np.random.Generator) -> None:
    indices = rng.choice(dataframe.index.to_numpy(), size=14, replace=False)
    multipliers = rng.uniform(4.0, 7.5, size=len(indices))
    dataframe.loc[indices, "processing_hours"] = (
        dataframe.loc[indices, "processing_hours"].to_numpy() * multipliers
    ).round(2)


def main() -> None:
    dataframe = generate_sample()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(OUTPUT_PATH, index=False, encoding="utf-8", lineterminator="\n")
    print(f"Wrote {len(dataframe)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
