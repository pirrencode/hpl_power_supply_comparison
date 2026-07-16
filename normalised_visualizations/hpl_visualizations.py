#!/usr/bin/env python3
"""Create route-wise Hyperloop power-supply comparison charts from output.csv.

Usage:
    python hpl_visualizations.py
    python hpl_visualizations.py --input output.csv --output-dir hpl_visualizations

Each chart places the electricity scenarios (solar, nuclear and fusion) on the
x-axis. Lines represent route_a, route_b and route_c. The additional
``route_median`` line is the median across all available routes for each energy
scenario.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROUTE_ORDER = ["route_a", "route_b", "route_c"]
SCENARIO_ORDER = ["solar", "nuclear", "fusion"]

METRICS = [
    ("capacity_factor_realised", "Realised capacity factor", "capacity_factor_realised.png", True),
    ("generation_capacity_mw", "Installed or contracted generation capacity (MW)", "generation_capacity_mw.png", False),
    ("generation_mwh", "Electricity generated during the simulation period (MWh)", "generation_mwh.png", False),
    ("external_demand_mwh", "Hyperloop external electricity demand (MWh)", "external_demand_mwh.png", False),
    ("grid_import_share", "Grid-import share", "grid_import_share.png", True),
    ("electricity_service_ratio", "Electricity-service ratio", "electricity_service_ratio.png", True),
    ("energy_not_served_mwh", "Energy not served (MWh)", "energy_not_served_mwh.png", False),
    ("on_time_departure_ratio", "On-time departure ratio", "on_time_departure_ratio.png", True),
    ("total_cost_eur", "Total simulation-period cost (EUR)", "total_cost_eur.png", False),
    ("levelized_delivered_electricity_cost_eur_per_mwh", "Levelised delivered electricity cost (EUR/MWh)", "levelized_cost_eur_per_mwh.png", False),
    ("cost_eur_per_passenger_km", "Cost (EUR/passenger-km)", "cost_eur_per_passenger_km.png", False),
    ("lifecycle_emissions_kgco2e", "Lifecycle emissions (kgCO2e)", "lifecycle_emissions_kgco2e.png", False),
    ("emissions_gco2e_per_passenger_km", "Lifecycle emissions (gCO2e/passenger-km)", "emissions_gco2e_per_passenger_km.png", False),
    ("land_use_ha", "Additional power-system land requirement (ha)", "land_use_ha.png", False),
    ("land_use_ha_per_gwh_served", "Land use (ha/GWh served)", "land_use_ha_per_gwh_served.png", False),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="output.csv", help="Simulation output CSV")
    parser.add_argument("--output-dir", default="hpl_visualizations", help="Directory for PNG files")
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution")

    # Jupyter/IPython injects its own command-line arguments, including
    # ``-f <kernel.json>``. parse_known_args() accepts the script's arguments
    # and safely ignores those unrelated kernel arguments.
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(f"Ignoring Jupyter/IPython arguments: {unknown_args}")
    return args


def select_scenario_rows(df: pd.DataFrame) -> pd.DataFrame:
    priorities = ["ANNUAL_SCENARIO", "SIMULATION_PERIOD_SCENARIO"]
    for period_type in priorities:
        selected = df[df["period_type"] == period_type].copy()
        if not selected.empty:
            return selected
    raise ValueError(
        "No ANNUAL_SCENARIO or SIMULATION_PERIOD_SCENARIO records were found in output.csv."
    )


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} was not found.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, low_memory=False)
    required = {"period_type", "route_id", "scenario"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"output.csv is missing required columns: {sorted(missing)}")

    annual = select_scenario_rows(df)
    annual["route_id"] = annual["route_id"].astype(str).str.strip().str.lower()
    annual["scenario"] = annual["scenario"].astype(str).str.strip().str.lower()
    annual = annual[
        annual["route_id"].isin(ROUTE_ORDER)
        & annual["scenario"].isin(SCENARIO_ORDER)
    ].copy()

    if annual.empty:
        raise ValueError("No route_a/route_b/route_c scenario records were found.")

    routes = [route for route in ROUTE_ORDER if route in annual["route_id"].unique()]
    scenarios = [scenario for scenario in SCENARIO_ORDER if scenario in annual["scenario"].unique()]

    generated = []
    median_rows = []

    for metric, ylabel, filename, display_as_percent in METRICS:
        if metric not in annual.columns:
            continue

        pivot = annual.pivot_table(
            index="scenario", columns="route_id", values=metric, aggfunc="median"
        ).reindex(index=scenarios, columns=routes)
        if pivot.empty or pivot.isna().all().all():
            continue

        plot_values = pivot.copy()
        axis_label = ylabel
        if display_as_percent:
            plot_values *= 100.0
            axis_label = f"{ylabel} (%)"

        route_median = plot_values.median(axis=1, skipna=True)

        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        for route in routes:
            ax.plot(
                scenarios,
                plot_values[route].to_numpy(dtype=float),
                marker="o",
                linewidth=2.0,
                label=route,
            )
        ax.plot(
            scenarios,
            route_median.to_numpy(dtype=float),
            marker="D",
            linestyle="--",
            linewidth=2.5,
            label="route_median",
        )

        ax.set_xlabel("Power-supply scenario")
        ax.set_ylabel(axis_label)
        ax.set_title(f"{ylabel}: comparison across routes")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
        fig.tight_layout()

        chart_path = output_dir / safe_filename(filename)
        fig.savefig(chart_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        generated.append((metric, chart_path.name))

        for scenario in scenarios:
            median_rows.append({
                "metric": metric,
                "scenario": scenario,
                "route_median": float(route_median.loc[scenario])
                if scenario in route_median.index and not pd.isna(route_median.loc[scenario])
                else np.nan,
                "display_as_percent": display_as_percent,
            })

    if not generated:
        raise ValueError("None of the configured visualisation metrics were found in output.csv.")

    pd.DataFrame(generated, columns=["metric", "filename"]).to_csv(
        output_dir / "visualization_index.csv", index=False
    )
    pd.DataFrame(median_rows).to_csv(
        output_dir / "route_median_values.csv", index=False
    )

    print(f"Created {len(generated)} graphs in: {output_dir.resolve()}")
    for _, filename in generated:
        print(f"- {filename}")


if __name__ == "__main__":
    main()
