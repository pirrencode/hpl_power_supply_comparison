"""
System dynamics model. 

Author: Aleksejs Vesjolijs

Required file:
    output.csv

Optional file:
    input.csv

Outputs:
    system_dynamics_model.png
    system_dynamics_model_scores.csv
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
    parser.add_argument(
        "--zoom-y-min",
        type=float,
        default=None,
        help=(
            "Optional lower limit for the focused vertical axis, e.g. 0.25. "
            "By default it is calculated from the observed P05-P95 range."
        ),
    )
    parser.add_argument(
        "--zoom-y-max",
        type=float,
        default=None,
        help=(
            "Optional upper limit for the focused vertical axis. "
            "By default it is calculated from the observed P05-P95 range."
        ),
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=30,
        help=(
            "Window for the rolling-average graph. Default: 30 periods; "
            "automatically reduced for shorter simulations."
        ),
    )

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



def scenario_frame(scores: pd.DataFrame, scenario: str) -> pd.DataFrame:

    return (
        scores[scores["scenario"] == scenario]
        .sort_values("time_period")
        .reset_index(drop=True)
    )


def calculate_zoom_limits(
    scores: pd.DataFrame,
    user_y_min: float | None,
    user_y_max: float | None,
) -> tuple[float, float]:

    lower = float(scores["average_normalised_score_p05"].min())
    upper = float(scores["average_normalised_score_p95"].max())
    span = max(upper - lower, 0.02)
    padding = max(0.015, span * 0.08)

    auto_min = max(0.0, lower - padding)
    auto_max = min(1.0, upper + padding)

    y_min = auto_min if user_y_min is None else float(user_y_min)
    y_max = auto_max if user_y_max is None else float(user_y_max)

    if not 0.0 <= y_min < y_max <= 1.0:
        raise ValueError(
            "Zoom-axis limits must satisfy 0 <= y_min < y_max <= 1."
        )

    return y_min, y_max


def add_scenario_lines(
    ax,
    scores: pd.DataFrame,
    *,
    use_rolling: bool = False,
    rolling_window: int = 30,
) -> None:
    for scenario in SCENARIO_ORDER:
        data = scenario_frame(scores, scenario)

        if use_rolling:
            central = data["average_normalised_score"].rolling(
                rolling_window,
                min_periods=1,
            ).mean()
            lower = data["average_normalised_score_p05"].rolling(
                rolling_window,
                min_periods=1,
            ).mean()
            upper = data["average_normalised_score_p95"].rolling(
                rolling_window,
                min_periods=1,
            ).mean()
            label = f"{scenario}_avg"
        else:
            central = data["average_normalised_score"]
            lower = data["average_normalised_score_p05"]
            upper = data["average_normalised_score_p95"]
            label = f"{scenario}_avg"

        line, = ax.plot(
            data["time_period"],
            central,
            linewidth=2.2,
            label=label,
        )
        ax.fill_between(
            data["time_period"].to_numpy(),
            lower.to_numpy(),
            upper.to_numpy(),
            alpha=0.18,
            color=line.get_color(),
        )


def decorate_time_axis(ax, final_period: int) -> None:
    ax.set_xlim(0, final_period)
    ax.set_xlabel("Simulation time period, t")
    ax.grid(True, alpha=0.3)


def plot_zoomed_scores(
    scores: pd.DataFrame,
    figure_path: Path,
    dpi: int,
    y_min: float | None,
    y_max: float | None,
) -> tuple[float, float]:
    fig, ax = plt.subplots(figsize=(12.0, 7.0))
    add_scenario_lines(ax, scores)

    final_period = int(scores["time_period"].max())
    zoom_min, zoom_max = calculate_zoom_limits(scores, y_min, y_max)

    decorate_time_axis(ax, final_period)
    ax.set_ylim(zoom_min, zoom_max)
    ax.set_ylabel("Average normalised performance score")
    ax.legend(title="Scenario average", loc="best")

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return zoom_min, zoom_max


def plot_full_scale_scores(
    scores: pd.DataFrame,
    figure_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 7.0))
    add_scenario_lines(ax, scores)

    final_period = int(scores["time_period"].max())
    decorate_time_axis(ax, final_period)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Average normalised performance score")
    ax.legend(title="Scenario average", loc="best")

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_rolling_scores(
    scores: pd.DataFrame,
    figure_path: Path,
    dpi: int,
    rolling_window: int,
    y_min: float | None,
    y_max: float | None,
) -> int:
    final_period = int(scores["time_period"].max())
    effective_window = max(1, min(int(rolling_window), final_period or 1))

    rolling_frames = []
    for scenario in SCENARIO_ORDER:
        data = scenario_frame(scores, scenario).copy()
        for column in [
            "average_normalised_score",
            "average_normalised_score_p05",
            "average_normalised_score_p95",
        ]:
            data[column] = data[column].rolling(
                effective_window,
                min_periods=1,
            ).mean()
        rolling_frames.append(data)
    rolling_scores = pd.concat(rolling_frames, ignore_index=True)

    fig, ax = plt.subplots(figsize=(12.0, 7.0))
    add_scenario_lines(
        ax,
        scores,
        use_rolling=True,
        rolling_window=effective_window,
    )

    zoom_min, zoom_max = calculate_zoom_limits(
        rolling_scores,
        y_min,
        y_max,
    )

    decorate_time_axis(ax, final_period)
    ax.set_ylim(zoom_min, zoom_max)
    ax.set_ylabel("Rolling average normalised performance score")
    ax.legend(title="Scenario average", loc="best")

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return effective_window


def build_relative_to_solar(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate nuclear and fusion performance differences relative to solar.

    Conservative uncertainty bounds are:
        lower = comparison P05 - solar P95
        upper = comparison P95 - solar P05
    """
    pivot = scores.pivot(
        index="time_period",
        columns="scenario",
        values=[
            "average_normalised_score",
            "average_normalised_score_p05",
            "average_normalised_score_p95",
        ],
    )

    records = []
    for comparison in ["nuclear", "fusion"]:
        for time_period in pivot.index:
            central = (
                pivot.loc[
                    time_period,
                    ("average_normalised_score", comparison),
                ]
                - pivot.loc[
                    time_period,
                    ("average_normalised_score", "solar"),
                ]
            )
            lower = (
                pivot.loc[
                    time_period,
                    ("average_normalised_score_p05", comparison),
                ]
                - pivot.loc[
                    time_period,
                    ("average_normalised_score_p95", "solar"),
                ]
            )
            upper = (
                pivot.loc[
                    time_period,
                    ("average_normalised_score_p95", comparison),
                ]
                - pivot.loc[
                    time_period,
                    ("average_normalised_score_p05", "solar"),
                ]
            )

            records.append(
                {
                    "time_period": int(time_period),
                    "scenario": comparison,
                    "relative_to_solar": float(central),
                    "relative_to_solar_p05": float(min(lower, central)),
                    "relative_to_solar_p95": float(max(upper, central)),
                }
            )

    return pd.DataFrame(records)


def plot_cumulative_advantage(
    scores: pd.DataFrame,
    figure_path: Path,
    dpi: int,
) -> pd.DataFrame:
    """
    Plot cumulative normalised-performance advantage relative to solar.

    Positive values indicate cumulative performance above solar; negative
    values indicate cumulative performance below solar.
    """
    relative = build_relative_to_solar(scores)
    cumulative_frames = []

    fig, ax = plt.subplots(figsize=(12.0, 7.0))

    for scenario in ["nuclear", "fusion"]:
        data = (
            relative[relative["scenario"] == scenario]
            .sort_values("time_period")
            .copy()
        )

        data["cumulative_advantage"] = data[
            "relative_to_solar"
        ].cumsum()
        data["cumulative_advantage_p05"] = data[
            "relative_to_solar_p05"
        ].cumsum()
        data["cumulative_advantage_p95"] = data[
            "relative_to_solar_p95"
        ].cumsum()

        line, = ax.plot(
            data["time_period"],
            data["cumulative_advantage"],
            linewidth=2.2,
            label=f"{scenario}_vs_solar",
        )
        ax.fill_between(
            data["time_period"].to_numpy(),
            data["cumulative_advantage_p05"].to_numpy(),
            data["cumulative_advantage_p95"].to_numpy(),
            alpha=0.18,
            color=line.get_color(),
        )
        cumulative_frames.append(data)

    final_period = int(scores["time_period"].max())
    decorate_time_axis(ax, final_period)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_ylabel(
        "Cumulative normalised-performance advantage relative to solar"
    )
    ax.set_title(
        "Cumulative scenario advantage relative to the solar baseline"
    )
    ax.legend(title="Comparison", loc="best")

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return pd.concat(cumulative_frames, ignore_index=True)


def main() -> None:
    args = parse_args()

    output_input_path = Path(args.output_input)
    model_input_path = Path(args.model_input)
    primary_figure_path = Path(args.figure)
    scores_path = Path(args.scores_output)

    output_dir = primary_figure_path.parent
    stem = primary_figure_path.stem
    suffix = primary_figure_path.suffix or ".png"

    full_scale_path = output_dir / f"{stem}_full_scale{suffix}"
    rolling_path = output_dir / f"{stem}_rolling_average{suffix}"
    cumulative_path = output_dir / f"{stem}_cumulative_advantage{suffix}"
    cumulative_csv_path = output_dir / (
        f"{stem}_cumulative_advantage.csv"
    )

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

    zoom_min, zoom_max = plot_zoomed_scores(
        scores,
        primary_figure_path,
        args.dpi,
        args.zoom_y_min,
        args.zoom_y_max,
    )
    plot_full_scale_scores(
        scores,
        full_scale_path,
        args.dpi,
    )
    effective_window = plot_rolling_scores(
        scores,
        rolling_path,
        args.dpi,
        args.rolling_window,
        args.zoom_y_min,
        args.zoom_y_max,
    )
    cumulative = plot_cumulative_advantage(
        scores,
        cumulative_path,
        args.dpi,
    )
    cumulative.to_csv(
        cumulative_csv_path,
        index=False,
        float_format="%.6f",
    )

    print(f"Created focused-scale graph: {primary_figure_path}")
    print(f"Created full-scale graph: {full_scale_path}")
    print(f"Created rolling-average graph: {rolling_path}")
    print(f"Created cumulative-advantage graph: {cumulative_path}")
    print(f"Created score data: {scores_path}")
    print(f"Created cumulative-advantage data: {cumulative_csv_path}")
    print(
        f"Dynamic horizon: t0 to t{max(periods)}; "
        f"focused vertical range: {zoom_min:.3f}-{zoom_max:.3f}; "
        f"rolling window: {effective_window}; "
        f"characteristics used: {len(usable_metrics)}."
    )


if __name__ == "__main__":
    main()
