"""
Create two combined comparison figures from Hyperloop simulation output:

1. A two-level annotated heatmap containing normalised scores and raw values.
2. A bubble scatter plot combining cost, realised capacity factor, generation,
   grid-import share and land requirement.

Required file:
    output.csv

Default outputs:
    normalised_visualizations/
        normalised_heatmap.png
        bubble_scatter.png
        normalised_heatmap_data.csv
        bubble_scatter_data.csv

The script works both from a terminal and inside Jupyter/IPython.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROUTE_ORDER = ["route_a", "route_b", "route_c"]
SCENARIO_ORDER = ["solar", "nuclear", "fusion"]

METRICS = {
    "land_use_ha": {
        "label": "Power-system land\nrequirement",
        "unit": "ha",
        "higher_is_better": False,
        "raw_format": lambda x: f"{x:,.2f} ha",
    },
    "grid_import_share": {
        "label": "Grid import\nshare",
        "unit": "%",
        "higher_is_better": False,
        "raw_format": lambda x: f"{100.0*x:,.1f}%",
    },
    "generation_mwh": {
        "label": "Electricity generated\nin simulation period",
        "unit": "MWh",
        "higher_is_better": True,
        "raw_format": lambda x: f"{x:,.0f} MWh",
    },
    "cost_eur_per_passenger_km": {
        "label": "Cost",
        "unit": "EUR/passenger-km",
        "higher_is_better": False,
        "raw_format": lambda x: f"{x:,.4f}\nEUR/pax-km",
    },
    "capacity_factor_realised": {
        "label": "Realised capacity\nfactor",
        "unit": "%",
        "higher_is_better": True,
        "raw_format": lambda x: f"{100.0*x:,.1f}%",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="output.csv",
        help="Simulation output CSV. Default: output.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="normalised_visualizations",
        help="Directory for figures and derived CSV files.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution. Default: 300",
    )
    parser.add_argument(
        "--include-route-median",
        action="store_true",
        help="Add route_median rows to the heatmap.",
    )

    # Jupyter passes arguments such as ``-f kernel.json``. Ignore arguments
    # that are not defined by this script.
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring Jupyter/IPython arguments: {unknown}")
    return args


def find_summary_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Select one summary row for each route and energy scenario."""
    preferred_types = [
        "ANNUAL_SCENARIO",
        "SIMULATION_PERIOD_SCENARIO",
    ]

    selected = pd.DataFrame()
    for period_type in preferred_types:
        candidate = df[df["period_type"].astype(str) == period_type].copy()
        if not candidate.empty:
            selected = candidate
            break

    if selected.empty:
        raise ValueError(
            "No ANNUAL_SCENARIO or SIMULATION_PERIOD_SCENARIO records were "
            "found in output.csv."
        )

    required = {
        "route_name",
        "scenario",
        *METRICS.keys(),
    }
    missing = sorted(required.difference(selected.columns))
    if missing:
        raise ValueError(f"output.csv is missing required columns: {missing}")

    selected["route_name"] = (
        selected["route_name"].astype(str).str.strip().str.lower()
    )
    selected["scenario"] = (
        selected["scenario"].astype(str).str.strip().str.lower()
    )

    selected = selected[
        selected["route_name"].isin(ROUTE_ORDER)
        & selected["scenario"].isin(SCENARIO_ORDER)
    ].copy()

    if selected.empty:
        raise ValueError(
            "No route_a, route_b or route_c summary records were found."
        )

    for column in METRICS:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")

    missing_values = selected[list(METRICS)].isna()
    if missing_values.any().any():
        bad = missing_values.stack()
        bad = bad[bad].index.tolist()
        raise ValueError(f"Missing or non-numeric metric values: {bad[:10]}")

    selected = (
        selected.sort_values(["route_name", "scenario"])
        .drop_duplicates(["route_name", "scenario"], keep="first")
    )

    expected = {(r, s) for r in ROUTE_ORDER for s in SCENARIO_ORDER}
    actual = set(zip(selected["route_name"], selected["scenario"]))
    absent = sorted(expected.difference(actual))
    if absent:
        warnings.warn(
            "Some route-scenario combinations are absent: "
            + ", ".join(f"{r}/{s}" for r, s in absent)
        )

    return selected


def minmax_score(values: pd.Series, higher_is_better: bool) -> pd.Series:
    """Return 0-1 scores; higher values always mean better performance."""
    minimum = values.min()
    maximum = values.max()

    if np.isclose(maximum, minimum):
        score = pd.Series(0.5, index=values.index, dtype=float)
    else:
        score = (values - minimum) / (maximum - minimum)

    if not higher_is_better:
        score = 1.0 - score

    return score.clip(0.0, 1.0)


def build_heatmap_frame(
    summary: pd.DataFrame,
    include_route_median: bool,
) -> pd.DataFrame:
    frame = summary.copy()

    if include_route_median:
        medians = (
            frame.groupby("scenario", as_index=False)[list(METRICS)]
            .median(numeric_only=True)
        )
        medians["route_name"] = "route_median"
        frame = pd.concat([frame, medians], ignore_index=True, sort=False)

    route_order = ROUTE_ORDER + (["route_median"] if include_route_median else [])
    frame["route_name"] = pd.Categorical(
        frame["route_name"],
        categories=route_order,
        ordered=True,
    )
    frame["scenario"] = pd.Categorical(
        frame["scenario"],
        categories=SCENARIO_ORDER,
        ordered=True,
    )
    frame = frame.sort_values(["route_name", "scenario"]).reset_index(drop=True)
    frame["row_label"] = (
        frame["route_name"].astype(str)
        + " — "
        + frame["scenario"].astype(str)
    )

    for metric, config in METRICS.items():
        frame[f"{metric}_score"] = minmax_score(
            frame[metric],
            config["higher_is_better"],
        )

    return frame


def plot_annotated_heatmap(
    frame: pd.DataFrame,
    output_path: Path,
    dpi: int,
) -> None:
    metric_names = list(METRICS)
    matrix = frame[[f"{m}_score" for m in metric_names]].to_numpy(dtype=float)

    fig_width = 12.5
    fig_height = max(7.0, 0.72 * len(frame) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_xticklabels(
        [METRICS[m]["label"] for m in metric_names],
        fontsize=10,
    )
    ax.set_yticks(np.arange(len(frame)))
    ax.set_yticklabels(frame["row_label"], fontsize=10)

    ax.set_xlabel("Performance indicator", labelpad=12)
    ax.set_ylabel("Route and power-supply scenario", labelpad=12)
    ax.set_title(
        "Normalised comparison of Hyperloop power-supply scenarios",
        pad=18,
    )

    # Cell borders.
    ax.set_xticks(np.arange(-0.5, len(metric_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(frame), 1), minor=True)
    ax.grid(which="minor", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Separate route groups visually.
    for index in range(3, len(frame), 3):
        ax.axhline(index - 0.5, linewidth=2.0)

    # Two-level annotation: score on first line; raw value below.
    for row_index, row in frame.iterrows():
        for column_index, metric in enumerate(metric_names):
            score = float(row[f"{metric}_score"])
            raw = float(row[metric])
            raw_text = METRICS[metric]["raw_format"](raw)
            annotation = f"Score {score:.2f}\n{raw_text}"

            text_colour = "white" if score < 0.25 or score > 0.72 else "black"
            ax.text(
                column_index,
                row_index,
                annotation,
                ha="center",
                va="center",
                fontsize=8.6,
                color=text_colour,
            )

    colour_bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colour_bar.set_label(
        "Normalised performance score\n(1 = most favourable)",
        rotation=90,
        labelpad=12,
    )

    fig.text(
        0.5,
        0.015,
        "Land requirement, grid import share and cost are reverse-normalised; "
        "electricity generation and realised capacity factor are positively normalised.",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    fig.tight_layout(rect=(0.0, 0.045, 1.0, 1.0))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def scale_bubble_sizes(values: pd.Series) -> np.ndarray:
    """Scale generation values to readable marker areas."""
    values = values.astype(float)
    minimum = values.min()
    maximum = values.max()

    if np.isclose(minimum, maximum):
        return np.full(len(values), 900.0)

    scaled = (values - minimum) / (maximum - minimum)
    return 350.0 + scaled * 1800.0


def plot_bubble_scatter(
    summary: pd.DataFrame,
    output_path: Path,
    dpi: int,
) -> None:
    frame = summary.copy().sort_values(["scenario", "route_name"])
    frame["bubble_size"] = scale_bubble_sizes(frame["generation_mwh"])

    route_markers = {
        "route_a": "o",
        "route_b": "s",
        "route_c": "^",
    }

    scenario_colours = {
        scenario: plt.get_cmap("tab10")(index)
        for index, scenario in enumerate(SCENARIO_ORDER)
    }

    fig, ax = plt.subplots(figsize=(12.0, 8.0))

    # Grid-import share controls transparency. High grid reliance is more opaque.
    grid_min = frame["grid_import_share"].min()
    grid_max = frame["grid_import_share"].max()

    def alpha_for_grid(value: float) -> float:
        if np.isclose(grid_min, grid_max):
            return 0.75
        normalised = (value - grid_min) / (grid_max - grid_min)
        return float(0.38 + 0.52 * normalised)

    for _, row in frame.iterrows():
        alpha = alpha_for_grid(float(row["grid_import_share"]))
        ax.scatter(
            row["cost_eur_per_passenger_km"],
            row["capacity_factor_realised"],
            s=row["bubble_size"],
            marker=route_markers[row["route_name"]],
            facecolor=scenario_colours[row["scenario"]],
            edgecolor="black",
            linewidth=max(0.8, min(3.5, 0.8 + row["land_use_ha"] / max(frame["land_use_ha"].max(), 1e-9) * 2.7)),
            alpha=alpha,
        )

        ax.annotate(
            f"{row['route_name']}\n"
            f"land={row['land_use_ha']:.2f} ha\n"
            f"grid={100.0*row['grid_import_share']:.1f}%",
            (
                row["cost_eur_per_passenger_km"],
                row["capacity_factor_realised"],
            ),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Cost (EUR/passenger-km)")
    ax.set_ylabel("Realised capacity factor")
    ax.set_title(
        "Cost, realised capacity, generation, grid reliance and land requirement"
    )
    ax.grid(True, alpha=0.3)

    # Scenario-colour legend.
    scenario_handles = [
        Patch(
            facecolor=scenario_colours[scenario],
            edgecolor="black",
            label=scenario,
        )
        for scenario in SCENARIO_ORDER
    ]
    scenario_legend = ax.legend(
        handles=scenario_handles,
        title="Power source",
        loc="upper left",
    )
    ax.add_artist(scenario_legend)

    # Route-marker legend.
    route_handles = [
        Line2D(
            [0],
            [0],
            marker=route_markers[route],
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=9,
            label=route,
        )
        for route in ROUTE_ORDER
    ]
    route_legend = ax.legend(
        handles=route_handles,
        title="Route",
        loc="upper right",
    )
    ax.add_artist(route_legend)

    # Bubble-size legend based on generation quantiles.
    quantiles = frame["generation_mwh"].quantile([0.1, 0.5, 0.9]).to_numpy()
    size_handles = []
    for value in quantiles:
        size = scale_bubble_sizes(
            pd.Series(
                [
                    frame["generation_mwh"].min(),
                    value,
                    frame["generation_mwh"].max(),
                ]
            )
        )[1]
        size_handles.append(
            ax.scatter(
                [],
                [],
                s=size,
                facecolor="none",
                edgecolor="black",
                label=f"{value:,.0f} MWh",
            )
        )
    ax.legend(
        handles=size_handles,
        title="Electricity generated\n(bubble area)",
        loc="lower right",
        fontsize=8,
    )

    fig.text(
        0.5,
        0.015,
        "Bubble area represents electricity generated; colour represents power source; "
        "marker shape represents route; opacity represents grid-import share; "
        "outline width and labels indicate land requirement.",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    fig.tight_layout(rect=(0.0, 0.045, 1.0, 1.0))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} was not found. Put output.csv in the current "
            "directory or pass --input <path>."
        )

    data = pd.read_csv(input_path, low_memory=False)
    summary = find_summary_rows(data)

    heatmap_frame = build_heatmap_frame(
        summary,
        include_route_median=args.include_route_median,
    )

    heatmap_path = output_dir / "normalised_heatmap.png"
    bubble_path = output_dir / "bubble_scatter.png"
    heatmap_csv = output_dir / "normalised_heatmap_data.csv"
    bubble_csv = output_dir / "bubble_scatter_data.csv"

    plot_annotated_heatmap(
        heatmap_frame,
        output_path=heatmap_path,
        dpi=args.dpi,
    )
    plot_bubble_scatter(
        summary,
        output_path=bubble_path,
        dpi=args.dpi,
    )

    heatmap_frame.to_csv(heatmap_csv, index=False)
    summary.to_csv(bubble_csv, index=False)

    print("Created:")
    print(f"  {heatmap_path}")
    print(f"  {bubble_path}")
    print(f"  {heatmap_csv}")
    print(f"  {bubble_csv}")


if __name__ == "__main__":
    main()
