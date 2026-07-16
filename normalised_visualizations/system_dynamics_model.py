"""
Create a dynamic, normalised system-performance graph from Hyperloop simulation
results.

Required file:
    output.csv

Optional file:
    input.csv  (used only to validate whether the horizon is 10 or 365 days)

Outputs:
    system_dynamics_model.png
    system_dynamics_model_scores.csv

For every time period, the script creates three composite trajectories:
solar_avg, nuclear_avg and fusion_avg. Each trajectory is the arithmetic mean of
route-specific performance indicators normalised to 0-1, where 1 always means
more favourable performance.

The default indicators are those for which the simulation output provides daily
Monte Carlo P05 and P95 values:

Higher is better:
    generation_mwh
    electricity_served_mwh
    capacity_factor_realised
    on_time_departure_ratio

Lower is better:
    energy_not_served_mwh
    grid_import_mwh
    levelized_delivered_electricity_cost_eur_per_mwh
    lifecycle_emissions_kgco2e

Normalisation is performed separately for each route and indicator using the
complete simulated horizon and all three power-supply scenarios. This prevents
long routes from dominating shorter routes while preserving temporal dynamics.
The shaded interval is an aggregated P05-P95 envelope obtained by applying the
same normalisation to each metric's Monte Carlo bounds and then averaging across
routes and indicators.

Time t0 is included by copying the first evaluated system state. The first daily
simulation result is t1 and the final state is tn.

The script is safe to run from a terminal, Jupyter Notebook or IPython.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIO_ORDER = ["solar", "nuclear", "fusion"]
ROUTE_ORDER = ["route_a", "route_b", "route_c"]

# True: higher is better. False: lower is better.
DEFAULT_METRICS = {
    "generation_mwh": True,
    "electricity_served_mwh": True,
    "energy_not_served_mwh": False,
    "grid_import_mwh": False,
    "capacity_factor_realised": True,
    "on_time_departure_ratio": True,
    "levelized_delivered_electricity_cost_eur_per_mwh": False,
    "lifecycle_emissions_kgco2e": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-input",
        default="output.csv",
        help="Simulation output CSV. Default: output.csv",
    )
    parser.add_argument(
        "--model-input",
        default="input.csv",
        help=(
            "Optional model input CSV used to validate the number of WEATHER "
            "periods. Default: input.csv"
        ),
    )
    parser.add_argument(
        "--figure",
        default="system_dynamics_model.png",
        help="Output figure. Default: system_dynamics_model.png",
    )
    parser.add_argument(
        "--scores-output",
        default="system_dynamics_model_scores.csv",
        help=(
            "Output CSV containing dynamic composite scores. "
            "Default: system_dynamics_model_scores.csv"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure resolution. Default: 300",
    )

    # Jupyter injects its own arguments, including -f <kernel.json>.
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring Jupyter/IPython arguments: {unknown}")
    return args


def select_daily_rows(data: pd.DataFrame) -> pd.DataFrame:
    required = {"period_type", "time_period", "scenario"}
    if "route_id" not in data.columns and "route_name" not in data.columns:
        required.add("route_name")
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"output.csv is missing required columns: {missing}")

    daily = data[
        data["period_type"].astype(str).str.strip() == "DAILY_SCENARIO"
    ].copy()
    if daily.empty:
        raise ValueError("No DAILY_SCENARIO records were found in output.csv.")

    route_source = "route_id" if "route_id" in daily.columns else "route_name"
    daily["route_key"] = (
        daily[route_source].astype(str).str.strip().str.lower()
    )
    daily["scenario"] = (
        daily["scenario"].astype(str).str.strip().str.lower()
    )
    daily["time_period"] = pd.to_numeric(daily["time_period"], errors="coerce")
    if daily["time_period"].isna().any():
        raise ValueError("DAILY_SCENARIO rows contain invalid time_period values.")
    daily["time_period"] = daily["time_period"].astype(int)

    daily = daily[
        daily["route_key"].isin(ROUTE_ORDER)
        & daily["scenario"].isin(SCENARIO_ORDER)
    ].copy()
    if daily.empty:
        raise ValueError(
            "No route_a, route_b or route_c DAILY_SCENARIO rows were found."
        )
    return daily


def validate_horizon_from_input(input_path: Path, observed_last_period: int) -> None:
    """Validate tn against WEATHER rows when input.csv is available."""
    if not input_path.exists():
        warnings.warn(
            f"{input_path} was not found; tn was inferred from output.csv."
        )
        return

    model_input = pd.read_csv(input_path, low_memory=False)
    if "record_type" not in model_input.columns or "day" not in model_input.columns:
        warnings.warn(
            "input.csv lacks record_type/day columns; horizon validation skipped."
        )
        return

    weather = model_input[
        model_input["record_type"].astype(str).str.upper() == "WEATHER"
    ].copy()
    if weather.empty:
        warnings.warn("input.csv contains no WEATHER rows; validation skipped.")
        return

    weather["day"] = pd.to_numeric(weather["day"], errors="coerce")
    weather = weather.dropna(subset=["day"])
    if weather.empty:
        warnings.warn("No numeric WEATHER days were found; validation skipped.")
        return

    input_last_period = int(weather["day"].max())
    if input_last_period != observed_last_period:
        raise ValueError(
            "Simulation horizon mismatch: input.csv contains "
            f"{input_last_period} WEATHER periods, while output.csv ends at "
            f"t{observed_last_period}."
        )


def validate_metrics(
    daily: pd.DataFrame,
    metrics: dict[str, bool],
) -> dict[str, bool]:
    """Retain only metrics with exact daily median, P05 and P95 columns."""
    usable: dict[str, bool] = {}
    excluded: list[str] = []

    for metric, higher_is_better in metrics.items():
        columns = [metric, f"{metric}_p05", f"{metric}_p95"]
        missing = [column for column in columns if column not in daily.columns]
        if missing:
            excluded.append(f"{metric}: missing {', '.join(missing)}")
            continue

        for column in columns:
            daily[column] = pd.to_numeric(daily[column], errors="coerce")

        if daily[columns].isna().all().any():
            excluded.append(f"{metric}: at least one column is entirely missing")
            continue

        usable[metric] = higher_is_better

    if excluded:
        warnings.warn(
            "Indicators without exact daily Monte Carlo bounds were excluded:\n- "
            + "\n- ".join(excluded)
        )
    if not usable:
        raise ValueError(
            "No indicators with daily median, P05 and P95 values are available."
        )
    return usable


def get_route_metric_bounds(
    daily: pd.DataFrame,
    metric: str,
) -> dict[str, tuple[float, float]]:
    """Create one fixed min-max scale for each route and indicator."""
    bounds: dict[str, tuple[float, float]] = {}
    for route in ROUTE_ORDER:
        route_rows = daily[daily["route_key"] == route]
        if route_rows.empty:
            continue

        values = pd.concat(
            [
                route_rows[metric],
                route_rows[f"{metric}_p05"],
                route_rows[f"{metric}_p95"],
            ],
            ignore_index=True,
        )
        values = pd.to_numeric(values, errors="coerce").dropna()
        if values.empty:
            continue
        bounds[route] = (float(values.min()), float(values.max()))
    return bounds


def normalise(
    value: float,
    minimum: float,
    maximum: float,
    higher_is_better: bool,
) -> float:
    """Transform a raw value to 0-1, with 1 always meaning better."""
    if pd.isna(value):
        return np.nan
    if np.isclose(maximum, minimum):
        return 0.5

    score = (float(value) - minimum) / (maximum - minimum)
    if not higher_is_better:
        score = 1.0 - score
    return float(np.clip(score, 0.0, 1.0))


def build_dynamic_scores(
    daily: pd.DataFrame,
    metrics: dict[str, bool],
) -> pd.DataFrame:
    all_bounds = {
        metric: get_route_metric_bounds(daily, metric)
        for metric in metrics
    }

    records: list[dict[str, object]] = []
    periods = sorted(daily["time_period"].unique())

    for period in periods:
        period_rows = daily[daily["time_period"] == period]

        for scenario in SCENARIO_ORDER:
            scenario_rows = period_rows[period_rows["scenario"] == scenario]

            central_scores: list[float] = []
            lower_scores: list[float] = []
            upper_scores: list[float] = []
            routes_used: set[str] = set()
            metrics_used: set[str] = set()

            for _, row in scenario_rows.iterrows():
                route = row["route_key"]
                if route not in ROUTE_ORDER:
                    continue

                for metric, higher_is_better in metrics.items():
                    if route not in all_bounds[metric]:
                        continue

                    minimum, maximum = all_bounds[metric][route]
                    median_raw = row[metric]
                    p05_raw = row[f"{metric}_p05"]
                    p95_raw = row[f"{metric}_p95"]
                    if pd.isna(median_raw) or pd.isna(p05_raw) or pd.isna(p95_raw):
                        continue

                    central = normalise(
                        median_raw, minimum, maximum, higher_is_better
                    )

                    if higher_is_better:
                        lower = normalise(
                            p05_raw, minimum, maximum, higher_is_better
                        )
                        upper = normalise(
                            p95_raw, minimum, maximum, higher_is_better
                        )
                    else:
                        # For lower-is-better metrics, raw P95 corresponds to
                        # the lower normalised-performance bound.
                        lower = normalise(
                            p95_raw, minimum, maximum, higher_is_better
                        )
                        upper = normalise(
                            p05_raw, minimum, maximum, higher_is_better
                        )

                    central_scores.append(central)
                    lower_scores.append(min(lower, upper))
                    upper_scores.append(max(lower, upper))
                    routes_used.add(route)
                    metrics_used.add(metric)

            if not central_scores:
                raise ValueError(
                    f"No valid normalised metrics for {scenario} at t{period}."
                )

            central_avg = float(np.mean(central_scores))
            p05_avg = float(np.mean(lower_scores))
            p95_avg = float(np.mean(upper_scores))

            # Numerical safeguard: the interval should contain its central line.
            p05_avg = min(p05_avg, central_avg)
            p95_avg = max(p95_avg, central_avg)

            records.append(
                {
                    "time_period": int(period),
                    "scenario": scenario,
                    "series_name": f"{scenario}_avg",
                    "average_normalised_score": central_avg,
                    "average_normalised_score_p05": p05_avg,
                    "average_normalised_score_p95": p95_avg,
                    "routes_used": len(routes_used),
                    "characteristics_used": len(metrics_used),
                }
            )

    scores = pd.DataFrame(records)

    # Include t0 by carrying back the first evaluated daily state.
    initial_rows = []
    for scenario in SCENARIO_ORDER:
        first = (
            scores[scores["scenario"] == scenario]
            .sort_values("time_period")
            .iloc[0]
            .copy()
        )
        first["time_period"] = 0
        initial_rows.append(first)

    scores = pd.concat([pd.DataFrame(initial_rows), scores], ignore_index=True)
    scores["time_period"] = scores["time_period"].astype(int)

    scenario_type = pd.CategoricalDtype(
        categories=SCENARIO_ORDER,
        ordered=True,
    )
    scores["scenario"] = scores["scenario"].astype(scenario_type)
    scores = scores.sort_values(["time_period", "scenario"]).reset_index(drop=True)
    scores["scenario"] = scores["scenario"].astype(str)
    return scores


def plot_scores(scores: pd.DataFrame, figure_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 7.0))

    for scenario in SCENARIO_ORDER:
        scenario_data = (
            scores[scores["scenario"] == scenario]
            .sort_values("time_period")
        )

        line, = ax.plot(
            scenario_data["time_period"],
            scenario_data["average_normalised_score"],
            linewidth=2.0,
            label=f"{scenario}_avg",
        )
        ax.fill_between(
            scenario_data["time_period"].to_numpy(),
            scenario_data["average_normalised_score_p05"].to_numpy(),
            scenario_data["average_normalised_score_p95"].to_numpy(),
            alpha=0.18,
            color=line.get_color(),
        )

    final_period = int(scores["time_period"].max())
    ax.set_xlim(0, final_period)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Simulation time period, t")
    ax.set_ylabel("Average normalised performance score")
    ax.set_title("Dynamic comparison of Hyperloop power-supply scenarios")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Scenario average", loc="best")

    ax.text(
        0.5,
        -0.13,
        "Lines show the mean normalised score across routes and performance "
        "characteristics; shaded areas show the aggregated Monte Carlo "
        "P05-P95 envelope.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    output_input_path = Path(args.output_input)
    model_input_path = Path(args.model_input)
    figure_path = Path(args.figure)
    scores_path = Path(args.scores_output)

    if not output_input_path.exists():
        raise FileNotFoundError(
            f"{output_input_path} was not found. Put output.csv in the current "
            "directory or pass --output-input <path>."
        )

    output_data = pd.read_csv(output_input_path, low_memory=False)
    daily = select_daily_rows(output_data)

    periods = sorted(daily["time_period"].unique())
    expected_periods = list(range(1, max(periods) + 1))
    if periods != expected_periods:
        raise ValueError(
            f"Daily periods must be consecutive from t1 to t{max(periods)}."
        )

    validate_horizon_from_input(
        model_input_path,
        observed_last_period=max(periods),
    )
    usable_metrics = validate_metrics(daily, DEFAULT_METRICS)
    scores = build_dynamic_scores(daily, usable_metrics)

    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(scores_path, index=False, float_format="%.6f")
    plot_scores(scores, figure_path, args.dpi)

    print(f"Created: {figure_path}")
    print(f"Created: {scores_path}")
    print(
        f"Dynamic horizon: t0 to t{max(periods)}; "
        f"series: solar_avg, nuclear_avg and fusion_avg; "
        f"characteristics used: {len(usable_metrics)}."
    )


if __name__ == "__main__":
    main()
