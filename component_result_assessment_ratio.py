"""
Create four assessment tables from Hyperloop simulation output.csv.

Outputs
-------
1. result_assessment.csv
   Contains the requested route-energy results and averages.

2. results_assessment_full.csv
   Contains the same values plus ratios relative to the applicable solar
   baseline.

3. comparative_analysis_solar_vs_nuclear.csv
   Comparative-analysis matrix C(m,k) for nuclear relative to solar.

4. comparative_analysis_solar_vs_fusion.csv
   Comparative-analysis matrix C(m,k) for fusion relative to solar.

Solar-baseline rules
--------------------
- route_a_nuclear and route_a_fusion are divided by route_a_solar.
- route_b_nuclear and route_b_fusion are divided by route_b_solar.
- route_c_nuclear and route_c_fusion are divided by route_c_solar.
- route_avg_nuclear and route_avg_fusion are divided by route_avg_solar.
- route_avg_for_all_energy_types is divided by route_avg_solar.
- Every solar row has ratio 1.0.

No ratios are created for route_length or time_periods.

If a solar baseline is zero:
- solar remains 1.0;
- another zero value is reported as 1.0;
- a positive value divided by zero is reported as inf.

Default input:
    output.csv

Default outputs:
    result_assessment.csv
    results_assessment_full.csv
    comparative_analysis_solar_vs_nuclear.csv
    comparative_analysis_solar_vs_fusion.csv

The script is safe to run from a terminal, Jupyter Notebook or IPython.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd


ROUTE_ORDER = ["route_a", "route_b", "route_c"]
SCENARIO_ORDER = ["solar", "nuclear", "fusion"]
COMPARATIVE_ROUTE_ORDER = ["route_a", "route_b", "route_c", "route_avg"]

SOURCE_TO_OUTPUT = {
    "route_length_km": "route_length",
    "time_period": "time_periods",
    "land_use_ha_per_gwh_served": "land_use_ha_gwh_served",
    "levelized_delivered_electricity_cost_eur_per_mwh":
        "levelised_delivered_electricity_cost",
    "cost_eur_per_passenger_km": "cost_passenger_km",
    "lifecycle_emissions_kgco2e": "lifecycle_emissions_co2",
    "total_cost_eur": "total_simulation_period_cost",
    "capacity_factor_realised": "relised_capacity_factor",
    "grid_import_share": "grid_import_share",
    "land_use_ha": "additional_power_system_land_requirement",
}

OUTPUT_COLUMNS = list(SOURCE_TO_OUTPUT.values())
NON_RATIO_COLUMNS = {"route_length", "time_periods"}
RATIO_SOURCE_COLUMNS = [
    column for column in OUTPUT_COLUMNS
    if column not in NON_RATIO_COLUMNS
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="output.csv",
        help="Simulation output file. Default: output.csv",
    )
    parser.add_argument(
        "--output",
        default="result_assessment.csv",
        help="Standard assessment output. Default: result_assessment.csv",
    )
    parser.add_argument(
        "--output-full",
        default="results_assessment_full.csv",
        help=(
            "Assessment output with solar-relative ratios. "
            "Default: results_assessment_full.csv"
        ),
    )
    parser.add_argument(
        "--output-solar-nuclear",
        default="comparative_analysis_solar_vs_nuclear.csv",
        help=(
            "C(m,k) table comparing nuclear with solar. "
            "Default: comparative_analysis_solar_vs_nuclear.csv"
        ),
    )
    parser.add_argument(
        "--output-solar-fusion",
        default="comparative_analysis_solar_vs_fusion.csv",
        help=(
            "C(m,k) table comparing fusion with solar. "
            "Default: comparative_analysis_solar_vs_fusion.csv"
        ),
    )

    # Jupyter injects arguments such as ``-f kernel.json``.
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring Jupyter/IPython arguments: {unknown}")
    return args


def select_scenario_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Select one summary record for every route and energy scenario."""
    preferred_period_types = [
        "ANNUAL_SCENARIO",
        "SIMULATION_PERIOD_SCENARIO",
    ]

    summary = pd.DataFrame()
    selected_period_type = None

    for period_type in preferred_period_types:
        candidate = df[
            df["period_type"].astype(str).str.strip() == period_type
        ].copy()
        if not candidate.empty:
            summary = candidate
            selected_period_type = period_type
            break

    if summary.empty:
        raise ValueError(
            "No ANNUAL_SCENARIO or SIMULATION_PERIOD_SCENARIO rows were "
            "found in output.csv."
        )

    required_columns = {
        "route_name",
        "scenario",
        *SOURCE_TO_OUTPUT.keys(),
    }
    missing_columns = sorted(required_columns.difference(summary.columns))
    if missing_columns:
        raise ValueError(
            "The simulation output is missing required columns: "
            f"{missing_columns}"
        )

    summary["route_name"] = (
        summary["route_name"].astype(str).str.strip().str.lower()
    )
    summary["scenario"] = (
        summary["scenario"].astype(str).str.strip().str.lower()
    )

    summary = summary[
        summary["route_name"].isin(ROUTE_ORDER)
        & summary["scenario"].isin(SCENARIO_ORDER)
    ].copy()

    for source_column in SOURCE_TO_OUTPUT:
        summary[source_column] = pd.to_numeric(
            summary[source_column],
            errors="coerce",
        )

    if summary[list(SOURCE_TO_OUTPUT)].isna().any().any():
        bad_locations = (
            summary[list(SOURCE_TO_OUTPUT)]
            .isna()
            .stack()
        )
        bad_locations = bad_locations[bad_locations].index.tolist()
        raise ValueError(
            "Missing or non-numeric values were found in required metrics: "
            f"{bad_locations[:10]}"
        )

    summary = (
        summary.sort_values(["route_name", "scenario"])
        .drop_duplicates(["route_name", "scenario"], keep="first")
    )

    expected = {
        (route, scenario)
        for route in ROUTE_ORDER
        for scenario in SCENARIO_ORDER
    }
    actual = set(zip(summary["route_name"], summary["scenario"]))
    missing_combinations = sorted(expected.difference(actual))

    if missing_combinations:
        raise ValueError(
            "The following route-energy combinations are missing: "
            + ", ".join(
                f"{route}_{scenario}"
                for route, scenario in missing_combinations
            )
        )

    print(f"Using period type: {selected_period_type}")
    return summary


def create_assessment(summary: pd.DataFrame) -> pd.DataFrame:
    """Build the requested 13-row assessment table."""
    renamed = summary.rename(columns=SOURCE_TO_OUTPUT).copy()
    rows = []

    # Nine route-energy rows.
    for route in ROUTE_ORDER:
        for scenario in SCENARIO_ORDER:
            selected = renamed[
                (renamed["route_name"] == route)
                & (renamed["scenario"] == scenario)
            ]

            if len(selected) != 1:
                raise ValueError(
                    f"Expected one row for {route}_{scenario}, "
                    f"but found {len(selected)}."
                )

            result_row = {
                "assessment_case": f"{route}_{scenario}",
            }
            result_row.update(
                selected.iloc[0][OUTPUT_COLUMNS].to_dict()
            )
            rows.append(result_row)

    # Arithmetic averages across the three routes for each energy type.
    for scenario in SCENARIO_ORDER:
        scenario_rows = renamed[renamed["scenario"] == scenario]
        average_values = scenario_rows[OUTPUT_COLUMNS].mean(numeric_only=True)

        result_row = {
            "assessment_case": f"route_avg_{scenario}",
        }
        result_row.update(average_values.to_dict())
        rows.append(result_row)

    # Arithmetic average across all nine route-energy combinations.
    overall_average = renamed[OUTPUT_COLUMNS].mean(numeric_only=True)
    result_row = {
        "assessment_case": "route_avg_for_all_energy_types",
    }
    result_row.update(overall_average.to_dict())
    rows.append(result_row)

    result = pd.DataFrame(
        rows,
        columns=["assessment_case", *OUTPUT_COLUMNS],
    )

    # Keep time periods as integers where possible.
    rounded = result["time_periods"].round()
    if (result["time_periods"] - rounded).abs().max() < 1e-9:
        result["time_periods"] = rounded.astype(int)

    return result


def baseline_case_for(assessment_case: str) -> str:
    """Return the solar baseline row for an assessment row."""
    if assessment_case.startswith("route_a_"):
        return "route_a_solar"
    if assessment_case.startswith("route_b_"):
        return "route_b_solar"
    if assessment_case.startswith("route_c_"):
        return "route_c_solar"
    if assessment_case.startswith("route_avg_"):
        return "route_avg_solar"

    raise ValueError(
        f"No solar baseline rule is defined for {assessment_case!r}."
    )


def calculate_ratio(
    current_value: float,
    baseline_value: float,
    is_baseline_row: bool,
    ratio_column: str,
    assessment_case: str,
    baseline_case: str,
) -> float:
    """Calculate a direct ratio while handling a zero baseline explicitly."""
    if is_baseline_row:
        return 1.0

    if np.isclose(baseline_value, 0.0):
        if np.isclose(current_value, 0.0):
            warnings.warn(
                f"{ratio_column} for {assessment_case} is set to 1.0 because "
                f"both it and solar baseline {baseline_case} are zero."
            )
            return 1.0

        warnings.warn(
            f"{ratio_column} for {assessment_case} is infinite because "
            f"solar baseline {baseline_case} equals zero."
        )
        return float("inf") if current_value > 0 else float("-inf")

    return current_value / baseline_value


def create_full_assessment(assessment: pd.DataFrame) -> pd.DataFrame:
    """Add direct value ratios relative to the applicable solar baseline."""
    indexed = assessment.set_index("assessment_case", drop=False)
    full = assessment.copy()

    for column in RATIO_SOURCE_COLUMNS:
        ratio_column = f"{column}_ratio"
        ratios = []

        for _, row in full.iterrows():
            case = row["assessment_case"]
            baseline_case = baseline_case_for(case)
            baseline_value = float(indexed.loc[baseline_case, column])
            current_value = float(row[column])

            ratios.append(
                calculate_ratio(
                    current_value=current_value,
                    baseline_value=baseline_value,
                    is_baseline_row=(case == baseline_case),
                    ratio_column=ratio_column,
                    assessment_case=case,
                    baseline_case=baseline_case,
                )
            )

        full[ratio_column] = ratios

    # Keep route_length and time_periods first. For every remaining metric,
    # place its ratio column immediately after the actual value column.
    ordered_columns = [
        "assessment_case",
        "route_length",
        "time_periods",
    ]
    for column in RATIO_SOURCE_COLUMNS:
        ordered_columns.extend(
            [
                column,
                f"{column}_ratio",
            ]
        )

    return full[ordered_columns]


def calculate_relative_difference_pct(
    current_value: float,
    baseline_value: float,
    assessment_case: str,
    baseline_case: str,
    indicator: str,
) -> float:
    """Calculate the relative difference defined in the manuscript."""
    if np.isclose(baseline_value, 0.0):
        if np.isclose(current_value, 0.0):
            warnings.warn(
                f"Relative difference for {assessment_case}, {indicator}, "
                f"is set to 0.0% because both values are zero."
            )
            return 0.0

        warnings.warn(
            f"Relative difference for {assessment_case}, {indicator}, "
            f"is infinite because solar baseline {baseline_case} equals zero."
        )
        return float("inf") if current_value > 0 else float("-inf")

    return (current_value - baseline_value) / baseline_value * 100.0


def create_comparative_matrix(
    assessment: pd.DataFrame,
    comparison_scenario: str,
) -> pd.DataFrame:
    """
    Create a long-format comparative-analysis table C(m,k).

    Indicator m identifies the performance metric and route k identifies
    route_a, route_b, route_c or the arithmetic route average.
    """
    if comparison_scenario not in {"nuclear", "fusion"}:
        raise ValueError(
            "comparison_scenario must be either 'nuclear' or 'fusion'."
        )

    indexed = assessment.set_index("assessment_case")
    rows = []

    for route in COMPARATIVE_ROUTE_ORDER:
        solar_case = f"{route}_solar"
        comparison_case = f"{route}_{comparison_scenario}"

        if solar_case not in indexed.index:
            raise ValueError(f"Missing assessment row: {solar_case}")
        if comparison_case not in indexed.index:
            raise ValueError(f"Missing assessment row: {comparison_case}")

        for indicator in RATIO_SOURCE_COLUMNS:
            solar_value = float(indexed.loc[solar_case, indicator])
            comparison_value = float(
                indexed.loc[comparison_case, indicator]
            )

            ratio = calculate_ratio(
                current_value=comparison_value,
                baseline_value=solar_value,
                is_baseline_row=False,
                ratio_column=f"{indicator}_ratio",
                assessment_case=comparison_case,
                baseline_case=solar_case,
            )
            relative_difference_pct = calculate_relative_difference_pct(
                current_value=comparison_value,
                baseline_value=solar_value,
                assessment_case=comparison_case,
                baseline_case=solar_case,
                indicator=indicator,
            )

            rows.append(
                {
                    "indicator_m": indicator,
                    "route_k": route,
                    "reference_scenario": "solar",
                    "comparison_scenario": comparison_scenario,
                    "solar_value": solar_value,
                    "comparison_value": comparison_value,
                    "C_m_k_ratio": ratio,
                    "relative_difference_pct": relative_difference_pct,
                }
            )

    result = pd.DataFrame(
        rows,
        columns=[
            "indicator_m",
            "route_k",
            "reference_scenario",
            "comparison_scenario",
            "solar_value",
            "comparison_value",
            "C_m_k_ratio",
            "relative_difference_pct",
        ],
    )

    indicator_order = {
        name: position for position, name in enumerate(RATIO_SOURCE_COLUMNS)
    }
    route_order = {
        name: position for position, name in enumerate(COMPARATIVE_ROUTE_ORDER)
    }
    result["_indicator_order"] = result["indicator_m"].map(indicator_order)
    result["_route_order"] = result["route_k"].map(route_order)
    result = (
        result.sort_values(["_indicator_order", "_route_order"])
        .drop(columns=["_indicator_order", "_route_order"])
        .reset_index(drop=True)
    )

    return result


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    full_output_path = Path(args.output_full)
    solar_nuclear_path = Path(args.output_solar_nuclear)
    solar_fusion_path = Path(args.output_solar_fusion)

    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} was not found. Put output.csv in the current "
            "directory or pass --input <path>."
        )

    data = pd.read_csv(input_path, low_memory=False)
    summary = select_scenario_summary(data)
    assessment = create_assessment(summary)
    full_assessment = create_full_assessment(assessment)
    solar_vs_nuclear = create_comparative_matrix(
        assessment,
        comparison_scenario="nuclear",
    )
    solar_vs_fusion = create_comparative_matrix(
        assessment,
        comparison_scenario="fusion",
    )

    for path in [
        output_path,
        full_output_path,
        solar_nuclear_path,
        solar_fusion_path,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    assessment.to_csv(output_path, index=False)
    full_assessment.to_csv(full_output_path, index=False)
    solar_vs_nuclear.to_csv(solar_nuclear_path, index=False)
    solar_vs_fusion.to_csv(solar_fusion_path, index=False)

    print(f"Created: {output_path}")
    print(f"Created: {full_output_path}")
    print(f"Created: {solar_nuclear_path}")
    print(f"Created: {solar_fusion_path}")
    print(f"Assessment rows: {len(assessment)}")
    print(
        "Rows in each comparative C(m,k) table: "
        f"{len(solar_vs_nuclear)}"
    )


if __name__ == "__main__":
    main()
