# -----------------------------------------------------------------------------
# Author: Aleksejs Vesjolijs
# -----------------------------------------------------------------------------

import math
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

INPUT_FILE = "input.csv"
OUTPUT_FILE = "output.csv"
COMPARISON_FILE = "route_comparison.csv"
SUMMARY_FILE = "annual_route_scenario_summary.csv"

SCENARIOS = ["solar", "nuclear", "fusion"]
ROUTE_ORDER = ["route_a", "route_b", "route_c"]

input_path = Path(INPUT_FILE)
if not input_path.exists():
    raise FileNotFoundError(
        f"{INPUT_FILE} was not found. Put input.csv in the same directory as the notebook."
    )

raw = pd.read_csv(input_path)
required_columns = {
    "record_type", "route_id", "route_name", "country_region", "scope",
    "scenario", "station_id", "station_name", "from_station_id",
    "to_station_id", "distance_km", "barren_soil_km", "parameter",
    "unit", "min_value", "mode_value", "max_value", "fixed_value",
    "notes", "source_url"
}
missing = required_columns.difference(raw.columns)
if missing:
    raise ValueError(f"input.csv is missing required columns: {sorted(missing)}")

text_columns = [
    "record_type", "route_id", "route_name", "country_region", "scope",
    "scenario", "station_name", "parameter", "unit", "notes", "source_url"
]
for col in text_columns:
    raw[col] = raw[col].fillna("").astype(str).str.strip()

numeric_columns = [
    "station_id", "from_station_id", "to_station_id", "distance_km",
    "barren_soil_km", "min_value", "mode_value", "max_value", "fixed_value"
]
for col in numeric_columns:
    raw[col] = pd.to_numeric(raw[col], errors="coerce")

required_weather_columns = {
    "profile_id", "route_id", "day", "season",
    "solar_generation_coefficient", "solar_price_coefficient",
    "solar_land_coefficient", "nuclear_generation_coefficient",
    "nuclear_price_coefficient", "nuclear_land_coefficient",
    "fusion_generation_coefficient", "fusion_price_coefficient",
    "fusion_land_coefficient", "grid_price_coefficient"
}
missing_weather = required_weather_columns.difference(raw.columns)
if missing_weather:
    raise ValueError(
        f"input.csv is missing required weather columns: {sorted(missing_weather)}"
    )

weather = raw[raw["record_type"].str.upper() == "WEATHER"].copy()
if weather.empty:
    raise ValueError(
        "No WEATHER records were found in input.csv. "
        "Use input_365.csv or input_10_ideal.csv as input.csv."
    )

weather["profile_id"] = (
    weather["profile_id"].fillna("input_profile").astype(str).str.strip()
)
weather["route_id"] = (
    weather["route_id"].fillna("").astype(str).str.strip().str.lower()
)
weather["season"] = weather["season"].fillna("").astype(str).str.strip()
weather["day"] = pd.to_numeric(weather["day"], errors="coerce")

weather_numeric_columns = [
    c for c in required_weather_columns if c.endswith("_coefficient")
]
for col in weather_numeric_columns:
    weather[col] = pd.to_numeric(weather[col], errors="coerce")

if weather["day"].isna().any():
    raise ValueError("input.csv contains non-numeric WEATHER day values.")
weather["day"] = weather["day"].astype(int)

if weather[weather_numeric_columns].isna().any().any():
    raise ValueError(
        "input.csv contains missing or non-numeric WEATHER coefficients."
    )
if (weather[weather_numeric_columns] <= 0).any().any():
    raise ValueError("All WEATHER coefficients in input.csv must be positive.")

parameter_rows = raw[raw["record_type"].str.upper() == "PARAMETER"].copy()
segment_rows = raw[raw["record_type"].str.upper() == "SEGMENT"].copy()
if segment_rows.empty:
    raise ValueError("No SEGMENT records were found in input.csv.")


def build_spec(row):
    return {
        "fixed": row["fixed_value"],
        "min": row["min_value"],
        "mode": row["mode_value"],
        "max": row["max_value"],
        "unit": row["unit"],
        "notes": row["notes"],
        "source_url": row["source_url"],
    }


def merge_specs(base, override):
    merged = dict(base)
    merged.update(override)
    return merged


global_specs = {}
scenario_default_specs = {}
route_specs = {}
route_scenario_specs = {}
route_station_specs = {}
route_station_names = {}
route_metadata = {}

for _, row in parameter_rows.iterrows():
    scope = row["scope"].upper()
    parameter = row["parameter"]
    if not parameter:
        continue
    spec = build_spec(row)
    route_id = row["route_id"].lower()

    if scope == "GLOBAL":
        global_specs[parameter] = spec
    elif scope == "ROUTE":
        if not route_id:
            raise ValueError(f"Route parameter '{parameter}' has no route_id.")
        route_specs.setdefault(route_id, {})[parameter] = spec
        route_metadata.setdefault(route_id, {})
        route_metadata[route_id]["route_name"] = row["route_name"] or route_id
        route_metadata[route_id]["country_region"] = row["country_region"]
    elif scope == "SCENARIO":
        scenario = row["scenario"].lower()
        if scenario not in SCENARIOS:
            continue
        if route_id:
            route_scenario_specs.setdefault((route_id, scenario), {})[parameter] = spec
            route_metadata.setdefault(route_id, {})
            route_metadata[route_id]["route_name"] = row["route_name"] or route_id
            route_metadata[route_id]["country_region"] = row["country_region"]
        else:
            scenario_default_specs.setdefault(scenario, {})[parameter] = spec
    elif scope == "STATION":
        if not route_id:
            raise ValueError(f"Station parameter '{parameter}' has no route_id.")
        if pd.isna(row["station_id"]):
            raise ValueError(f"Station parameter '{parameter}' has no station_id.")
        sid = int(row["station_id"])
        route_station_specs.setdefault((route_id, sid), {})[parameter] = spec
        route_station_names[(route_id, sid)] = row["station_name"] or f"Station {sid}"
        route_metadata.setdefault(route_id, {})
        route_metadata[route_id]["route_name"] = row["route_name"] or route_id
        route_metadata[route_id]["country_region"] = row["country_region"]

for scenario in SCENARIOS:
    if scenario not in scenario_default_specs:
        raise ValueError(f"Default scenario '{scenario}' is missing from input.csv.")

segments_by_route = {}
for _, row in segment_rows.iterrows():
    route_id = row["route_id"].lower()
    if not route_id:
        raise ValueError("Every SEGMENT row must contain route_id.")
    if pd.isna(row["from_station_id"]) or pd.isna(row["to_station_id"]):
        raise ValueError("Every SEGMENT row must contain from_station_id and to_station_id.")
    if pd.isna(row["distance_km"]) or row["distance_km"] <= 0:
        raise ValueError("Every SEGMENT row must contain a positive distance_km.")
    segments_by_route.setdefault(route_id, []).append({
        "from": int(row["from_station_id"]),
        "to": int(row["to_station_id"]),
        "distance_km": float(row["distance_km"]),
        "barren_soil_km": 0.0 if pd.isna(row["barren_soil_km"]) else float(row["barren_soil_km"]),
    })
    route_metadata.setdefault(route_id, {})
    route_metadata[route_id]["route_name"] = row["route_name"] or route_id
    route_metadata[route_id]["country_region"] = row["country_region"]

route_ids = [rid for rid in ROUTE_ORDER if rid in segments_by_route]
route_ids += sorted(set(segments_by_route).difference(route_ids))
if len(route_ids) < 2:
    raise ValueError("At least two routes are required for route comparison.")

def beta_pert_sample(spec, rng, size=None, lam=4.0):
    fixed = spec.get("fixed", np.nan)
    if not pd.isna(fixed):
        if size is None:
            return float(fixed)
        return np.full(size, float(fixed), dtype=float)

    a = spec.get("min", np.nan)
    m = spec.get("mode", np.nan)
    b = spec.get("max", np.nan)
    if pd.isna(a) or pd.isna(m) or pd.isna(b):
        raise ValueError(f"Incomplete parameter specification: {spec}")
    a, m, b = float(a), float(m), float(b)
    if b < a or not (a <= m <= b):
        raise ValueError(f"Invalid Beta-PERT values: min={a}, mode={m}, max={b}")
    if math.isclose(a, b):
        if size is None:
            return a
        return np.full(size, a, dtype=float)

    alpha = 1.0 + lam * (m - a) / (b - a)
    beta = 1.0 + lam * (b - m) / (b - a)
    draw = rng.beta(alpha, beta, size=size)
    return a + draw * (b - a)


def sample_parameter(specs, name, rng, lam, size=None, default=None):
    if name not in specs:
        if default is not None:
            return default
        raise KeyError(f"Required parameter '{name}' is missing from input.csv.")
    return beta_pert_sample(specs[name], rng, size=size, lam=lam)


def sample_all(specs, rng, lam, excluded=None):
    excluded = set(excluded or [])
    return {
        name: beta_pert_sample(spec, rng, lam=lam)
        for name, spec in specs.items()
        if name not in excluded
    }


def capital_recovery_factor(rate, years):
    rate = float(rate)
    years = float(years)
    if years <= 0:
        raise ValueError("Asset lifetime must be positive.")
    if math.isclose(rate, 0.0):
        return 1.0 / years
    return rate * (1.0 + rate) ** years / ((1.0 + rate) ** years - 1.0)


def q05(x):
    return float(np.nanpercentile(x, 5))


def q50(x):
    return float(np.nanpercentile(x, 50))


def q95(x):
    return float(np.nanpercentile(x, 95))


def safe_divide(a, b):
    if b is None or pd.isna(b) or math.isclose(float(b), 0.0):
        return np.nan
    return a / b


configured_simulation_days = int(round(global_specs["simulation_days"]["fixed"]))
mc_runs = int(round(global_specs["monte_carlo_runs"]["fixed"]))
random_seed = int(round(global_specs["random_seed"]["fixed"]))
pert_lambda = float(global_specs["beta_pert_lambda"]["fixed"])
if configured_simulation_days <= 0 or mc_runs <= 0:
    raise ValueError("simulation_days and monte_carlo_runs must be positive.")

weather_days_by_route = {}
for rid in route_ids:
    route_weather_days = sorted(weather.loc[weather["route_id"] == rid, "day"].unique())
    if not route_weather_days:
        raise ValueError(f"input.csv contains no WEATHER rows for {rid}.")
    expected_days = list(range(1, max(route_weather_days) + 1))
    if route_weather_days != expected_days:
        raise ValueError(
            f"input.csv: WEATHER days for {rid} must be consecutive from 1 to {max(route_weather_days)}."
        )
    if weather.loc[weather["route_id"] == rid, "day"].duplicated().any():
        raise ValueError(f"input.csv contains duplicate WEATHER day rows for {rid}.")
    weather_days_by_route[rid] = len(route_weather_days)

if len(set(weather_days_by_route.values())) != 1:
    raise ValueError(
        f"All routes must contain the same number of weather days: {weather_days_by_route}"
    )
simulation_days = next(iter(weather_days_by_route.values()))
weather_profiles = sorted(weather["profile_id"].dropna().unique())
weather_profile_id = weather_profiles[0] if len(weather_profiles) == 1 else "mixed_input_profile"
weather_lookup = {
    (row.route_id, int(row.day)): row
    for row in weather.itertuples(index=False)
    if row.route_id in route_ids
}

daily_metric_names = [
    "planned_pod_cycles", "completed_pod_cycles", "passengers_served",
    "passenger_km", "travel_time_hours_per_full_route",
    "route_average_speed_kmh", "maximum_pod_speed_kmh", "packs_per_pod",
    "battery_capacity_per_pod_kwh", "battery_energy_margin_ratio",
    "tube_pressure_pa", "external_energy_intensity_wh_per_pax_km",
    "fixed_infrastructure_share_effective", "external_demand_mwh",
    "electricity_served_mwh", "energy_not_served_mwh",
    "electricity_service_ratio", "generation_capacity_mw",
    "storage_capacity_mwh", "generation_mwh", "grid_import_mwh",
    "grid_export_mwh", "grid_import_share", "bess_charge_mwh",
    "bess_discharge_mwh", "bess_soc_mwh", "capacity_factor_realised",
    "solar_capacity_factor_realised",
    "sunny_day_indicator", "sunny_day_share_assumed",
    "weather_generation_coefficient", "weather_price_coefficient",
    "weather_land_coefficient", "grid_price_coefficient",
    "effective_grid_price_eur_per_mwh", "peak_demand_mw",
    "on_time_departure_ratio", "blocked_service_events",
    "charged_pack_availability", "min_ready_packs", "charger_utilisation",
    "swap_bay_utilisation", "battery_throughput_mwh",
    "battery_replacement_equivalent_packs", "generation_cost_eur",
    "generation_cost_eur_per_mwh", "grid_cost_eur", "common_station_cost_eur",
    "battery_replacement_cost_eur", "export_revenue_eur", "total_cost_eur",
    "levelized_delivered_electricity_cost_eur_per_mwh",
    "lifecycle_emissions_kgco2e", "land_use_ha",
    "generation_yield_mwh_per_mw_year", "land_use_ha_per_gwh_served",
    "land_use_ha_per_gwh_generated", "cost_eur_per_completed_cycle", "cost_eur_per_passenger_km",
    "emissions_gco2e_per_passenger_km", "loss_of_load_indicator",
    "cumulative_external_demand_mwh", "cumulative_energy_not_served_mwh",
    "cumulative_total_cost_eur", "cumulative_lifecycle_emissions_kgco2e",
]

station_annual_metric_names = [
    "external_demand_mwh", "electricity_served_mwh", "energy_not_served_mwh",
    "generation_mwh", "grid_import_mwh", "direct_services_completed",
    "swap_services_completed", "blocked_service_events", "minimum_ready_packs",
    "average_charger_utilisation", "average_swap_bay_utilisation",
    "on_time_departure_ratio", "allocated_total_cost_eur",
    "allocated_lifecycle_emissions_kgco2e",
]

all_output_rows = []
annual_summary_rows = []
rng_master = np.random.default_rng(random_seed)

for route_number, route_id in enumerate(route_ids):
    metadata = route_metadata.get(route_id, {})
    route_name = route_id
    country_region = metadata.get("country_region", "")
    segments = segments_by_route[route_id]

    station_ids = sorted(
        set(seg["from"] for seg in segments) | set(seg["to"] for seg in segments)
    )
    for sid in station_ids:
        if (route_id, sid) not in route_station_specs:
            raise ValueError(f"No STATION parameter records found for {route_id}, station {sid}.")

    station_names = {
        sid: route_station_names.get((route_id, sid), f"Station {sid}")
        for sid in station_ids
    }
    station_index = {sid: i for i, sid in enumerate(station_ids)}
    n_stations = len(station_ids)

    route_length_km = sum(seg["distance_km"] for seg in segments)
    segment_barren_soil_km = sum(seg["barren_soil_km"] for seg in segments)
    route_global_specs = merge_specs(global_specs, route_specs.get(route_id, {}))
    published_barren_spec = route_global_specs.get("published_route_barren_soil_coverage_km")
    if published_barren_spec is not None and not pd.isna(published_barren_spec.get("fixed", np.nan)):
        barren_soil_km = float(published_barren_spec["fixed"])
    else:
        barren_soil_km = segment_barren_soil_km
    barren_soil_share = barren_soil_km / route_length_km if route_length_km else np.nan
    route_weather_table = weather[weather["route_id"] == route_id].sort_values("day")
    route_weather_profile = route_weather_table["profile_id"].iloc[0]
    route_sunny_share = float(
        np.mean(route_weather_table["solar_generation_coefficient"].to_numpy() >= 1.0)
    )

    incident_segments = {sid: [] for sid in station_ids}
    for seg in segments:
        incident_segments[seg["from"]].append(seg)
        incident_segments[seg["to"]].append(seg)
    if any(len(incident_segments[sid]) == 0 for sid in station_ids):
        raise ValueError(f"Route {route_id} contains an isolated station.")

    # The adjacent-distance weights sum to one for both open and closed routes.
    node_weights = {
        sid: sum(seg["distance_km"] for seg in incident_segments[sid])
        / (2.0 * route_length_km)
        for sid in station_ids
    }

    daily_results = {
        scenario: {
            metric: np.full((mc_runs, simulation_days), np.nan, dtype=float)
            for metric in daily_metric_names
        }
        for scenario in SCENARIOS
    }
    station_annual_results = {
        scenario: {
            metric: np.full((mc_runs, n_stations), np.nan, dtype=float)
            for metric in station_annual_metric_names
        }
        for scenario in SCENARIOS
    }

    for run in range(mc_runs):
        # Same route-demand realisation is used for all three power-supply scenarios.
        run_seed = int(rng_master.integers(0, 2**32 - 1)) + route_number * 1000003
        rng = np.random.default_rng(run_seed)

        g = sample_all(
            route_global_specs,
            rng,
            pert_lambda,
            excluded={"simulation_days", "monte_carlo_runs", "random_seed", "beta_pert_lambda"},
        )

        cycles_per_direction_base = g["cycles_per_direction_per_day"]
        directions = int(round(g["number_of_directions"]))
        passengers_per_pod = g["passenger_equivalent_per_pod"]
        route_avg_speed = g["route_average_speed_kmh"]
        tube_pressure_pa = g["tube_pressure_pa"]
        prop_wh_base = g["propulsion_energy_wh_per_pax_km"]
        prop_wh = prop_wh_base * (
            route_avg_speed / max(1e-9, g["reference_speed_kmh"])
        ) ** g["propulsion_speed_exponent"]
        lev_wh = g["levitation_energy_wh_per_pax_km"]
        aux_wh = g["pod_aux_energy_wh_per_pax_km"]
        braking_share = g["braking_energy_share"]
        regen_eff = g["regenerative_recovery_efficiency"]
        fixed_share = g["fixed_infrastructure_share"] * (
            g["vacuum_pressure_reference_pa"] / max(1e-9, tube_pressure_pa)
        ) ** g["vacuum_load_pressure_exponent"]
        fixed_share = float(np.clip(fixed_share, 0.05, 0.60))
        peak_ratio = g["peak_to_average_power_ratio"]
        charge_eff = g["charging_efficiency"]
        conv_eff = g["conversion_distribution_efficiency"]
        swap_share = g["swap_service_share"]
        packs_per_pod = g["packs_per_pod"]
        pack_kwh = g["pack_nominal_capacity_kwh"]
        usable_dod = g["usable_depth_of_discharge"]
        battery_reserve = g["operational_battery_reserve"]
        cycle_life = g["battery_cycle_life_efc"]
        battery_eol_soh = g["battery_eol_soh"]
        calendar_deg = g["battery_calendar_degradation_per_year"]
        battery_capex = g["battery_capex_eur_per_kwh"]
        battery_embodied_ef = g["battery_embodied_emissions_kg_per_kwh"]
        station_aux_kwh_service = g["station_aux_energy_kwh_per_service"]
        weekend_factor = g["weekend_demand_factor"]
        discount_rate = g["discount_rate"]
        turnaround_min = g["terminal_turnaround_time_min"]
        minimum_headway_s = g["minimum_headway_s"]
        operation_hours = g["operation_hours_per_day"]
        grid_price = g["grid_price_eur_per_mwh"]
        grid_ef = g["grid_lifecycle_emissions_kg_per_mwh"]

        net_onboard_wh_per_pax_km = (
            prop_wh + lev_wh + aux_wh - prop_wh * braking_share * regen_eff
        )
        if "external_energy_intensity_override_wh_per_pax_km" in g:
            external_energy_intensity_wh_per_pax_km = g[
                "external_energy_intensity_override_wh_per_pax_km"
            ]
            net_onboard_wh_per_pax_km = (
                external_energy_intensity_wh_per_pax_km
                * charge_eff * conv_eff * (1.0 - fixed_share)
            )
        else:
            external_energy_intensity_wh_per_pax_km = (
                net_onboard_wh_per_pax_km / (charge_eff * conv_eff) / (1.0 - fixed_share)
            )
        if net_onboard_wh_per_pax_km <= 0:
            raise ValueError(f"Calculated onboard energy intensity is not positive for {route_id}.")

        segment_energy_kwh_per_pod = {
            (seg["from"], seg["to"]):
            net_onboard_wh_per_pax_km * passengers_per_pod * seg["distance_km"] / 1000.0
            for seg in segments
        }
        max_segment_energy_kwh = max(segment_energy_kwh_per_pod.values())
        usable_energy_per_pack_kwh = (
            pack_kwh * usable_dod * (1.0 - battery_reserve) * battery_eol_soh
        )
        required_packs_per_pod = int(
            math.ceil(max_segment_energy_kwh / max(1e-9, usable_energy_per_pack_kwh))
        )
        packs_per_pod = max(int(round(packs_per_pod)), required_packs_per_pod)
        usable_battery_kwh = packs_per_pod * usable_energy_per_pack_kwh
        battery_energy_margin_ratio = usable_battery_kwh / max_segment_energy_kwh

        full_route_travel_time_h = (
            route_length_km / route_avg_speed + n_stations * turnaround_min / 60.0
        )
        max_cycles_per_direction = operation_hours * 3600.0 / minimum_headway_s
        # The scheduled cycle rate is constrained by the sampled minimum headway.
        # This safeguard is silent because the input schedule is now configured to
        # remain feasible across the full route-specific headway range.
        cycles_per_direction_base = min(cycles_per_direction_base, max_cycles_per_direction)

        # Sample station parameters once per Monte Carlo run.
        st = {}
        for sid in station_ids:
            s_rng = np.random.default_rng(run_seed + sid * 7919)
            specs = route_station_specs[(route_id, sid)]
            st[sid] = sample_all(specs, s_rng, pert_lambda)

        station_capex_total = 0.0
        station_fixed_om_total = 0.0
        for sid in station_ids:
            p = st[sid]
            installed_pack_inventory = p["initial_ready_packs"] + p["initial_depleted_packs"]
            station_capex = (
                g["station_base_capex_eur"]
                + p["number_of_chargers"] * p["charger_rated_power_mw"]
                  * g["charger_capex_eur_per_mw"]
                + p["number_of_swap_bays"] * g["swap_bay_capex_eur"]
                + installed_pack_inventory * pack_kwh * battery_capex
            )
            station_capex_total += station_capex
            station_fixed_om_total += station_capex * g["station_fixed_om_fraction_per_year"]

        station_crf = capital_recovery_factor(discount_rate, g["station_asset_lifetime_years"])
        common_station_cost_per_day = (
            station_capex_total * station_crf + station_fixed_om_total
        ) / 365.0

        demand_factor_daily = beta_pert_sample(
            route_global_specs["daily_demand_factor"],
            rng,
            size=simulation_days,
            lam=pert_lambda,
        )
        for day_idx in range(simulation_days):
            if day_idx % 7 in (5, 6):
                demand_factor_daily[day_idx] *= weekend_factor

        for scenario in SCENARIOS:
            srng = np.random.default_rng(
                run_seed + {"solar": 101, "nuclear": 202, "fusion": 303}[scenario]
            )
            scenario_specs = merge_specs(
                scenario_default_specs[scenario],
                route_scenario_specs.get((route_id, scenario), {}),
            )
            sp = sample_all(scenario_specs, srng, pert_lambda)

            if scenario == "solar":
                generation_capacity_mw = sum(st[sid]["solar_capacity_mw"] for sid in station_ids)
                storage_capacity_mwh = sum(
                    st[sid]["stationary_bess_capacity_mwh"] for sid in station_ids
                )
            else:
                generation_capacity_mw = sp["generation_capacity_mw"]
                storage_capacity_mwh = 0.0

            gen_crf = capital_recovery_factor(discount_rate, sp["asset_lifetime_years"])
            gen_fixed_cost_per_day = (
                generation_capacity_mw * 1000.0 * sp["generation_capex_eur_per_kw"] * gen_crf
                + generation_capacity_mw * 1000.0 * sp["fixed_om_eur_per_kw_year"]
            ) / 365.0

            storage_fixed_cost_per_day = 0.0
            if scenario == "solar":
                storage_crf = capital_recovery_factor(
                    discount_rate, sp["storage_lifetime_years"]
                )
                storage_capex = storage_capacity_mwh * 1000.0 * sp["storage_capex_eur_per_kwh"]
                storage_fixed_cost_per_day = (
                    storage_capex * storage_crf
                    + storage_capex * sp["storage_fixed_om_fraction_per_year"]
                ) / 365.0

            if scenario == "solar":
                base_land_use_ha = (
                    generation_capacity_mw * sp["land_use_ha_per_mw"]
                    + storage_capacity_mwh * sp["storage_land_ha_per_mwh"]
                )
            else:
                base_land_use_ha = generation_capacity_mw * sp["land_use_ha_per_mw"]

            ready_packs = np.array(
                [st[sid]["initial_ready_packs"] for sid in station_ids], dtype=float
            )
            depleted_packs = np.array(
                [st[sid]["initial_depleted_packs"] for sid in station_ids], dtype=float
            )
            bess_soc = np.array(
                [
                    0.5 * st[sid]["stationary_bess_capacity_mwh"]
                    if scenario == "solar" else 0.0
                    for sid in station_ids
                ], dtype=float
            )

            sta = {
                "external_demand_mwh": np.zeros(n_stations),
                "electricity_served_mwh": np.zeros(n_stations),
                "energy_not_served_mwh": np.zeros(n_stations),
                "generation_mwh": np.zeros(n_stations),
                "grid_import_mwh": np.zeros(n_stations),
                "direct_services_completed": np.zeros(n_stations),
                "swap_services_completed": np.zeros(n_stations),
                "blocked_service_events": np.zeros(n_stations),
                "minimum_ready_packs": ready_packs.copy(),
                "charger_utilisation_sum": np.zeros(n_stations),
                "swap_bay_utilisation_sum": np.zeros(n_stations),
                "on_time_departure_ratio_sum": np.zeros(n_stations),
            }

            planned_outage_days = int(round(
                sp.get("planned_outage_days_per_year", 0.0)
                * simulation_days / 365.0
            ))
            if planned_outage_days > 0:
                max_start = max(1, simulation_days - planned_outage_days + 1)
                planned_outage_start = int(srng.integers(1, max_start + 1))
                planned_outage_end = planned_outage_start + planned_outage_days - 1
            else:
                planned_outage_start = -1
                planned_outage_end = -1

            cumulative_total_cost = 0.0
            cumulative_emissions = 0.0
            cumulative_external_demand = 0.0
            cumulative_energy_not_served = 0.0

            for day_idx in range(simulation_days):
                day = day_idx + 1
                weather_row = weather_lookup[(route_id, day)]
                weather_generation_coefficient = float(
                    getattr(weather_row, f"{scenario}_generation_coefficient")
                )
                weather_price_coefficient = float(
                    getattr(weather_row, f"{scenario}_price_coefficient")
                )
                weather_land_coefficient = float(
                    getattr(weather_row, f"{scenario}_land_coefficient")
                )
                grid_price_coefficient = float(weather_row.grid_price_coefficient)
                effective_grid_price = grid_price * grid_price_coefficient
                land_use_ha = base_land_use_ha * weather_land_coefficient

                unconstrained_cycles = cycles_per_direction_base * demand_factor_daily[day_idx]
                cycles_per_direction = min(unconstrained_cycles, max_cycles_per_direction)
                planned_pod_cycles = cycles_per_direction * directions

                # Station events differ for open and closed routes. Each incident segment
                # contributes one arrival stream to the station.
                service_events_by_station = np.array(
                    [cycles_per_direction * len(incident_segments[sid]) for sid in station_ids],
                    dtype=float,
                )
                direct_req_services = service_events_by_station * (1.0 - swap_share)
                swap_req_services = service_events_by_station * swap_share

                onboard_req_mwh = np.zeros(n_stations)
                fixed_ext_mwh = np.zeros(n_stations)
                station_aux_ext_mwh = np.zeros(n_stations)
                external_req_mwh = np.zeros(n_stations)
                connection_capacity_mwh = np.zeros(n_stations)
                charger_input_capacity_mwh = np.zeros(n_stations)
                swap_capacity_services = np.zeros(n_stations)

                total_onboard_req_mwh = 0.0
                for sid in station_ids:
                    i = station_index[sid]
                    incident_energy_kwh = sum(
                        segment_energy_kwh_per_pod[(seg["from"], seg["to"])]
                        for seg in incident_segments[sid]
                    )
                    onboard_req_mwh[i] = cycles_per_direction * incident_energy_kwh / 1000.0
                    total_onboard_req_mwh += onboard_req_mwh[i]

                external_traction_mwh = total_onboard_req_mwh / (charge_eff * conv_eff)
                total_fixed_ext_mwh = (
                    external_traction_mwh * fixed_share / max(1e-9, 1.0 - fixed_share)
                )

                for sid in station_ids:
                    i = station_index[sid]
                    p = st[sid]
                    availability = p["station_availability"]
                    fixed_ext_mwh[i] = total_fixed_ext_mwh * node_weights[sid]
                    station_aux_ext_mwh[i] = (
                        service_events_by_station[i] * station_aux_kwh_service
                        / 1000.0 / conv_eff
                    )
                    external_req_mwh[i] = (
                        onboard_req_mwh[i] / (charge_eff * conv_eff)
                        + fixed_ext_mwh[i] + station_aux_ext_mwh[i]
                    )
                    connection_capacity_mwh[i] = p["connection_limit_mw"] * 24.0 * availability
                    charger_input_capacity_mwh[i] = (
                        p["number_of_chargers"] * p["charger_rated_power_mw"]
                        * 24.0 * availability
                    )
                    swap_capacity_services[i] = (
                        p["number_of_swap_bays"] * 24.0 * 60.0
                        / p["battery_swap_time_min"] * availability
                    )

                effective_demand_mwh = np.minimum(external_req_mwh, connection_capacity_mwh)
                connection_unserved_mwh = np.maximum(0.0, external_req_mwh - effective_demand_mwh)

                generation_by_station = np.zeros(n_stations)
                grid_import_by_station = np.zeros(n_stations)
                grid_export_by_station = np.zeros(n_stations)
                served_by_station = np.zeros(n_stations)
                bess_charge_by_station = np.zeros(n_stations)
                bess_discharge_by_station = np.zeros(n_stations)
                capacity_factor_realised = np.nan
                solar_capacity_factor = np.nan
                sunny_day_indicator = np.nan
                sunny_day_share = np.nan

                if scenario == "solar":
                    sunny_day_share = route_sunny_share
                    sunny_day_indicator = 1.0 if weather_generation_coefficient >= 1.0 else 0.0
                    clear_sky_cf = sp.get(
                        "clear_sky_capacity_factor", sp.get("annual_capacity_factor", 0.20)
                    )
                    solar_capacity_factor = float(np.clip(
                        clear_sky_cf * weather_generation_coefficient,
                        0.0, 0.40
                    ))

                    deficits = np.zeros(n_stations)
                    for sid in station_ids:
                        i = station_index[sid]
                        p = st[sid]
                        generation_by_station[i] = p["solar_capacity_mw"] * 24.0 * solar_capacity_factor
                        direct_pv = min(generation_by_station[i], effective_demand_mwh[i])
                        remaining_demand = effective_demand_mwh[i] - direct_pv
                        surplus_pv = generation_by_station[i] - direct_pv

                        bess_capacity = p["stationary_bess_capacity_mwh"]
                        bess_power = p["stationary_bess_power_mw"]
                        max_bess_input = min(
                            surplus_pv,
                            bess_power * 24.0,
                            max(0.0, bess_capacity - bess_soc[i]) / sp["bess_charge_efficiency"],
                        )
                        bess_charge_by_station[i] = max_bess_input
                        bess_soc[i] += max_bess_input * sp["bess_charge_efficiency"]
                        surplus_pv -= max_bess_input

                        max_bess_output = min(
                            remaining_demand,
                            bess_power * 24.0,
                            bess_soc[i] * sp["bess_discharge_efficiency"],
                        )
                        bess_discharge_by_station[i] = max_bess_output
                        bess_soc[i] -= max_bess_output / sp["bess_discharge_efficiency"]
                        remaining_demand -= max_bess_output

                        deficits[i] = max(0.0, remaining_demand)
                        grid_export_by_station[i] = max(0.0, surplus_pv)
                        served_by_station[i] = direct_pv + max_bess_output

                    total_deficit = deficits.sum()
                    total_grid_import = min(total_deficit, sp["grid_backup_limit_mw"] * 24.0)
                    if total_deficit > 0:
                        grid_import_by_station = total_grid_import * deficits / total_deficit
                    served_by_station += grid_import_by_station

                else:
                    availability = sp["availability_factor"] * weather_generation_coefficient
                    if planned_outage_start <= day <= planned_outage_end:
                        availability *= sp["planned_outage_derate"]
                    if srng.random() < sp["forced_outage_probability_per_day"]:
                        availability *= sp["forced_outage_derate"]

                    if scenario == "nuclear":
                        net_fraction = 1.0 - sp["plant_auxiliary_fraction"]
                    else:
                        net_fraction = (
                            1.0 - sp["recirculating_power_fraction"]
                            - sp["plant_auxiliary_fraction"]
                        )
                    net_fraction = max(0.0, net_fraction)
                    total_generation_mwh = (
                        generation_capacity_mw * 24.0 * availability * net_fraction
                    )
                    total_effective_demand = effective_demand_mwh.sum()
                    direct_generation_used = min(total_generation_mwh, total_effective_demand)
                    if total_effective_demand > 0:
                        generation_by_station = (
                            direct_generation_used * effective_demand_mwh / total_effective_demand
                        )
                    deficit_after_generation = np.maximum(
                        0.0, effective_demand_mwh - generation_by_station
                    )
                    total_deficit = deficit_after_generation.sum()
                    total_grid_import = min(
                        total_deficit, sp["grid_backup_limit_mw"] * 24.0
                    )
                    if total_deficit > 0:
                        grid_import_by_station = (
                            total_grid_import * deficit_after_generation / total_deficit
                        )
                    served_by_station = generation_by_station + grid_import_by_station
                    surplus = max(0.0, total_generation_mwh - direct_generation_used)
                    if surplus > 0 and total_effective_demand > 0:
                        grid_export_by_station = (
                            surplus * effective_demand_mwh / total_effective_demand
                        )

                electricity_served_mwh = np.minimum(served_by_station, effective_demand_mwh)
                energy_not_served_by_station = (
                    np.maximum(0.0, effective_demand_mwh - electricity_served_mwh)
                    + connection_unserved_mwh
                )

                # Battery-service stocks and flows.
                direct_completed = np.zeros(n_stations)
                swap_completed = np.zeros(n_stations)
                blocked_services = np.zeros(n_stations)
                pack_availability = np.ones(n_stations)
                charger_utilisation = np.zeros(n_stations)
                swap_utilisation = np.zeros(n_stations)
                station_service_ratio = np.ones(n_stations)
                battery_throughput_by_station = np.zeros(n_stations)

                for sid in station_ids:
                    i = station_index[sid]
                    events = service_events_by_station[i]
                    charging_external_served = max(
                        0.0,
                        electricity_served_mwh[i] - fixed_ext_mwh[i] - station_aux_ext_mwh[i],
                    )
                    charger_input_available = min(
                        charging_external_served * conv_eff,
                        charger_input_capacity_mwh[i],
                    )
                    onboard_energy_deliverable = charger_input_available * charge_eff
                    average_energy_per_service_mwh = safe_divide(onboard_req_mwh[i], events)
                    if pd.isna(average_energy_per_service_mwh):
                        average_energy_per_service_mwh = 0.0

                    direct_energy_request = onboard_req_mwh[i] * (1.0 - swap_share)
                    direct_energy_delivered = min(onboard_energy_deliverable, direct_energy_request)
                    remaining_onboard_energy = max(0.0, onboard_energy_deliverable - direct_energy_delivered)

                    if average_energy_per_service_mwh > 0:
                        direct_completed[i] = min(
                            direct_req_services[i],
                            direct_energy_delivered / average_energy_per_service_mwh,
                        )
                    else:
                        direct_completed[i] = direct_req_services[i]

                    required_swap_packs = swap_req_services[i] * packs_per_pod
                    pack_availability[i] = min(
                        1.0,
                        safe_divide(ready_packs[i], required_swap_packs)
                        if required_swap_packs > 0 else 1.0,
                    )
                    max_swaps_from_ready_packs = safe_divide(ready_packs[i], packs_per_pod)
                    swap_completed[i] = min(
                        swap_req_services[i], swap_capacity_services[i], max_swaps_from_ready_packs
                    )
                    issued_packs = swap_completed[i] * packs_per_pod
                    ready_packs[i] -= issued_packs
                    depleted_packs[i] += issued_packs

                    average_energy_per_pack_mwh = safe_divide(
                        average_energy_per_service_mwh, packs_per_pod
                    )
                    if average_energy_per_pack_mwh and not pd.isna(average_energy_per_pack_mwh):
                        charged_packs_today = min(
                            depleted_packs[i],
                            remaining_onboard_energy / average_energy_per_pack_mwh,
                        )
                    else:
                        charged_packs_today = depleted_packs[i]
                    energy_used_offboard = charged_packs_today * (
                        average_energy_per_pack_mwh if average_energy_per_pack_mwh else 0.0
                    )
                    depleted_packs[i] -= charged_packs_today
                    ready_packs[i] += charged_packs_today

                    completed_services = direct_completed[i] + swap_completed[i]
                    blocked_services[i] = max(0.0, events - completed_services)
                    station_service_ratio[i] = min(
                        1.0, safe_divide(completed_services, events) if events > 0 else 1.0
                    )
                    charger_utilisation[i] = min(
                        1.0,
                        safe_divide(
                            direct_energy_delivered / charge_eff + energy_used_offboard / charge_eff,
                            charger_input_capacity_mwh[i],
                        ) if charger_input_capacity_mwh[i] > 0 else 0.0,
                    )
                    swap_utilisation[i] = min(
                        1.0,
                        safe_divide(swap_completed[i], swap_capacity_services[i])
                        if swap_capacity_services[i] > 0 else 0.0,
                    )
                    battery_throughput_by_station[i] = direct_energy_delivered + energy_used_offboard

                route_on_time_ratio = float(np.min(station_service_ratio))
                completed_pod_cycles = planned_pod_cycles * route_on_time_ratio
                passengers_served = completed_pod_cycles * passengers_per_pod
                passenger_km = passengers_served * route_length_km

                total_external_demand = external_req_mwh.sum()
                total_electricity_served = electricity_served_mwh.sum()
                total_ens = energy_not_served_by_station.sum()
                electricity_service_ratio = min(
                    1.0, safe_divide(total_electricity_served, total_external_demand)
                )
                if scenario == "solar":
                    total_generation = generation_by_station.sum()
                else:
                    total_generation = generation_by_station.sum() + grid_export_by_station.sum()

                capacity_factor_realised = safe_divide(
                    total_generation, generation_capacity_mw * 24.0
                )
                if not pd.isna(capacity_factor_realised):
                    capacity_factor_realised = float(np.clip(capacity_factor_realised, 0.0, 1.0))

                total_grid_import = grid_import_by_station.sum()
                total_grid_export = grid_export_by_station.sum()
                total_bess_charge = bess_charge_by_station.sum()
                total_bess_discharge = bess_discharge_by_station.sum()
                total_bess_soc = bess_soc.sum()
                peak_demand_mw = total_external_demand / 24.0 * peak_ratio
                total_battery_throughput = battery_throughput_by_station.sum()

                throughput_replacements = (
                    total_battery_throughput * 1000.0
                    / max(1e-9, pack_kwh * cycle_life * usable_dod)
                )
                total_pack_inventory = ready_packs.sum() + depleted_packs.sum()
                calendar_replacements = (
                    total_pack_inventory * calendar_deg / 365.0
                    / max(1e-9, 1.0 - battery_eol_soh)
                )
                replacement_equivalent_packs = throughput_replacements + calendar_replacements
                battery_replacement_cost = replacement_equivalent_packs * pack_kwh * battery_capex
                battery_replacement_emissions = (
                    replacement_equivalent_packs * pack_kwh * battery_embodied_ef
                )

                generation_variable_cost = total_generation * (
                    sp["variable_om_eur_per_mwh"] + sp["fuel_cost_eur_per_mwh"]
                ) * weather_price_coefficient
                generation_cost = (
                    gen_fixed_cost_per_day + storage_fixed_cost_per_day
                    + generation_variable_cost
                )
                grid_cost = total_grid_import * effective_grid_price
                export_revenue = (
                    total_grid_export * sp["export_price_eur_per_mwh"]
                    * grid_price_coefficient
                )
                total_cost = (
                    generation_cost + grid_cost + common_station_cost_per_day
                    + battery_replacement_cost - export_revenue
                )
                lifecycle_emissions = (
                    total_generation * sp["lifecycle_emissions_kg_per_mwh"]
                    + total_grid_import * grid_ef + battery_replacement_emissions
                )

                cumulative_total_cost += total_cost
                cumulative_emissions += lifecycle_emissions
                cumulative_external_demand += total_external_demand
                cumulative_energy_not_served += total_ens

                daily_values = {
                    "planned_pod_cycles": planned_pod_cycles,
                    "completed_pod_cycles": completed_pod_cycles,
                    "passengers_served": passengers_served,
                    "passenger_km": passenger_km,
                    "travel_time_hours_per_full_route": full_route_travel_time_h,
                    "route_average_speed_kmh": route_avg_speed,
                    "maximum_pod_speed_kmh": g["maximum_pod_speed_kmh"],
                    "packs_per_pod": packs_per_pod,
                    "battery_capacity_per_pod_kwh": packs_per_pod * pack_kwh,
                    "battery_energy_margin_ratio": battery_energy_margin_ratio,
                    "tube_pressure_pa": tube_pressure_pa,
                    "external_energy_intensity_wh_per_pax_km": external_energy_intensity_wh_per_pax_km,
                    "fixed_infrastructure_share_effective": fixed_share,
                    "external_demand_mwh": total_external_demand,
                    "electricity_served_mwh": total_electricity_served,
                    "energy_not_served_mwh": total_ens,
                    "electricity_service_ratio": electricity_service_ratio,
                    "generation_capacity_mw": generation_capacity_mw,
                    "storage_capacity_mwh": storage_capacity_mwh,
                    "generation_mwh": total_generation,
                    "grid_import_mwh": total_grid_import,
                    "grid_export_mwh": total_grid_export,
                    "grid_import_share": safe_divide(total_grid_import, total_external_demand),
                    "bess_charge_mwh": total_bess_charge,
                    "bess_discharge_mwh": total_bess_discharge,
                    "bess_soc_mwh": total_bess_soc,
                    "capacity_factor_realised": capacity_factor_realised,
                    "solar_capacity_factor_realised": solar_capacity_factor,
                    "sunny_day_indicator": sunny_day_indicator,
                    "sunny_day_share_assumed": sunny_day_share,
                    "weather_generation_coefficient": weather_generation_coefficient,
                    "weather_price_coefficient": weather_price_coefficient,
                    "weather_land_coefficient": weather_land_coefficient,
                    "grid_price_coefficient": grid_price_coefficient,
                    "effective_grid_price_eur_per_mwh": effective_grid_price,
                    "peak_demand_mw": peak_demand_mw,
                    "on_time_departure_ratio": route_on_time_ratio,
                    "blocked_service_events": blocked_services.sum(),
                    "charged_pack_availability": float(np.min(pack_availability)),
                    "min_ready_packs": float(np.min(ready_packs)),
                    "charger_utilisation": float(np.mean(charger_utilisation)),
                    "swap_bay_utilisation": float(np.mean(swap_utilisation)),
                    "battery_throughput_mwh": total_battery_throughput,
                    "battery_replacement_equivalent_packs": replacement_equivalent_packs,
                    "generation_cost_eur": generation_cost,
                    "generation_cost_eur_per_mwh": safe_divide(generation_cost, total_generation),
                    "grid_cost_eur": grid_cost,
                    "common_station_cost_eur": common_station_cost_per_day,
                    "battery_replacement_cost_eur": battery_replacement_cost,
                    "export_revenue_eur": export_revenue,
                    "total_cost_eur": total_cost,
                    "levelized_delivered_electricity_cost_eur_per_mwh": safe_divide(
                        total_cost, total_electricity_served
                    ),
                    "lifecycle_emissions_kgco2e": lifecycle_emissions,
                    "land_use_ha": land_use_ha,
                    "generation_yield_mwh_per_mw_year": safe_divide(total_generation, generation_capacity_mw),
                    "land_use_ha_per_gwh_served": safe_divide(land_use_ha, total_electricity_served / 1000.0),
                    "land_use_ha_per_gwh_generated": safe_divide(land_use_ha, total_generation / 1000.0),
                    "cost_eur_per_completed_cycle": safe_divide(total_cost, completed_pod_cycles),
                    "cost_eur_per_passenger_km": safe_divide(total_cost, passenger_km),
                    "emissions_gco2e_per_passenger_km": safe_divide(
                        lifecycle_emissions * 1000.0, passenger_km
                    ),
                    "loss_of_load_indicator": 1.0 if total_ens > 1e-9 else 0.0,
                    "cumulative_external_demand_mwh": cumulative_external_demand,
                    "cumulative_energy_not_served_mwh": cumulative_energy_not_served,
                    "cumulative_total_cost_eur": cumulative_total_cost,
                    "cumulative_lifecycle_emissions_kgco2e": cumulative_emissions,
                }
                for metric, value in daily_values.items():
                    daily_results[scenario][metric][run, day_idx] = value

                sta["external_demand_mwh"] += external_req_mwh
                sta["electricity_served_mwh"] += electricity_served_mwh
                sta["energy_not_served_mwh"] += energy_not_served_by_station
                sta["generation_mwh"] += generation_by_station
                sta["grid_import_mwh"] += grid_import_by_station
                sta["direct_services_completed"] += direct_completed
                sta["swap_services_completed"] += swap_completed
                sta["blocked_service_events"] += blocked_services
                sta["minimum_ready_packs"] = np.minimum(sta["minimum_ready_packs"], ready_packs)
                sta["charger_utilisation_sum"] += charger_utilisation
                sta["swap_bay_utilisation_sum"] += swap_utilisation
                sta["on_time_departure_ratio_sum"] += station_service_ratio

            annual_cost_run = np.nansum(daily_results[scenario]["total_cost_eur"][run, :])
            annual_emissions_run = np.nansum(
                daily_results[scenario]["lifecycle_emissions_kgco2e"][run, :]
            )
            served_weights = sta["electricity_served_mwh"] / max(
                1e-9, sta["electricity_served_mwh"].sum()
            )
            for metric in [
                "external_demand_mwh", "electricity_served_mwh",
                "energy_not_served_mwh", "generation_mwh", "grid_import_mwh",
                "direct_services_completed", "swap_services_completed",
                "blocked_service_events", "minimum_ready_packs"
            ]:
                station_annual_results[scenario][metric][run, :] = sta[metric]
            station_annual_results[scenario]["average_charger_utilisation"][run, :] = (
                sta["charger_utilisation_sum"] / simulation_days
            )
            station_annual_results[scenario]["average_swap_bay_utilisation"][run, :] = (
                sta["swap_bay_utilisation_sum"] / simulation_days
            )
            station_annual_results[scenario]["on_time_departure_ratio"][run, :] = (
                sta["on_time_departure_ratio_sum"] / simulation_days
            )
            station_annual_results[scenario]["allocated_total_cost_eur"][run, :] = (
                annual_cost_run * served_weights
            )
            station_annual_results[scenario]["allocated_lifecycle_emissions_kgco2e"][run, :] = (
                annual_emissions_run * served_weights
            )

    # -------------------------------------------------------------------------
    # 5. Aggregate one route into daily, annual and station rows
    # -------------------------------------------------------------------------
    daily_uncertainty_metrics = {
        "external_demand_mwh", "electricity_served_mwh", "energy_not_served_mwh",
        "generation_mwh", "grid_import_mwh", "capacity_factor_realised",
        "solar_capacity_factor_realised",
        "on_time_departure_ratio", "total_cost_eur",
        "levelized_delivered_electricity_cost_eur_per_mwh",
        "lifecycle_emissions_kgco2e", "cumulative_external_demand_mwh",
        "cumulative_energy_not_served_mwh", "cumulative_total_cost_eur",
        "cumulative_lifecycle_emissions_kgco2e",
    }

    for scenario in SCENARIOS:
        for day_idx in range(simulation_days):
            row = {
                "period_type": "DAILY_SCENARIO",
                "time_period": day_idx + 1,
                "weather_profile_id": route_weather_profile,
                "weather_season": weather_lookup[(route_id, day_idx + 1)].season,
                "route_id": route_id,
                "route_name": route_name,
                "country_region": country_region,
                "scenario": scenario,
                "station_id": "",
                "station_name": "",
                "route_length_km": route_length_km,
                "number_of_stations": n_stations,
                "barren_soil_km": barren_soil_km,
                "segment_barren_soil_sum_km": segment_barren_soil_km,
                "barren_soil_share": barren_soil_share,
                "monte_carlo_runs": mc_runs,
            }
            for metric in daily_metric_names:
                values = daily_results[scenario][metric][:, day_idx]
                row[metric] = q50(values) if not np.all(np.isnan(values)) else np.nan
                if metric in daily_uncertainty_metrics and not np.all(np.isnan(values)):
                    row[f"{metric}_p05"] = q05(values)
                    row[f"{metric}_p95"] = q95(values)
            all_output_rows.append(row)

        additive_metrics = {
            "planned_pod_cycles", "completed_pod_cycles", "passengers_served",
            "passenger_km", "external_demand_mwh", "electricity_served_mwh",
            "energy_not_served_mwh", "generation_mwh", "grid_import_mwh",
            "grid_export_mwh", "bess_charge_mwh", "bess_discharge_mwh",
            "blocked_service_events", "battery_throughput_mwh",
            "battery_replacement_equivalent_packs", "generation_cost_eur",
            "grid_cost_eur", "common_station_cost_eur",
            "battery_replacement_cost_eur", "export_revenue_eur", "total_cost_eur",
            "lifecycle_emissions_kgco2e", "loss_of_load_indicator",
            "sunny_day_indicator",
        }
        annual_run_values = {}
        for metric in daily_metric_names:
            arr = daily_results[scenario][metric]
            if metric in additive_metrics:
                annual_run_values[metric] = np.nansum(arr, axis=1)
            elif metric == "on_time_departure_ratio":
                annual_run_values[metric] = (
                    np.nansum(daily_results[scenario]["completed_pod_cycles"], axis=1)
                    / np.maximum(1e-9, np.nansum(
                        daily_results[scenario]["planned_pod_cycles"], axis=1
                    ))
                )
            elif metric == "electricity_service_ratio":
                annual_run_values[metric] = (
                    np.nansum(daily_results[scenario]["electricity_served_mwh"], axis=1)
                    / np.maximum(1e-9, np.nansum(
                        daily_results[scenario]["external_demand_mwh"], axis=1
                    ))
                )
            elif metric == "grid_import_share":
                annual_run_values[metric] = (
                    np.nansum(daily_results[scenario]["grid_import_mwh"], axis=1)
                    / np.maximum(1e-9, np.nansum(
                        daily_results[scenario]["external_demand_mwh"], axis=1
                    ))
                )
            elif metric == "cost_eur_per_completed_cycle":
                annual_run_values[metric] = (
                    np.nansum(daily_results[scenario]["total_cost_eur"], axis=1)
                    / np.maximum(1e-9, np.nansum(
                        daily_results[scenario]["completed_pod_cycles"], axis=1
                    ))
                )
            elif metric == "cost_eur_per_passenger_km":
                annual_run_values[metric] = (
                    np.nansum(daily_results[scenario]["total_cost_eur"], axis=1)
                    / np.maximum(1e-9, np.nansum(
                        daily_results[scenario]["passenger_km"], axis=1
                    ))
                )
            elif metric == "generation_cost_eur_per_mwh":
                annual_run_values[metric] = (
                    np.nansum(daily_results[scenario]["generation_cost_eur"], axis=1)
                    / np.maximum(1e-9, np.nansum(
                        daily_results[scenario]["generation_mwh"], axis=1
                    ))
                )
            elif metric == "levelized_delivered_electricity_cost_eur_per_mwh":
                annual_run_values[metric] = (
                    np.nansum(daily_results[scenario]["total_cost_eur"], axis=1)
                    / np.maximum(1e-9, np.nansum(
                        daily_results[scenario]["electricity_served_mwh"], axis=1
                    ))
                )
            elif metric == "emissions_gco2e_per_passenger_km":
                annual_run_values[metric] = (
                    np.nansum(daily_results[scenario]["lifecycle_emissions_kgco2e"], axis=1)
                    * 1000.0
                    / np.maximum(1e-9, np.nansum(
                        daily_results[scenario]["passenger_km"], axis=1
                    ))
                )
            elif metric in {"min_ready_packs", "charged_pack_availability"}:
                annual_run_values[metric] = np.nanmin(arr, axis=1)
            elif metric == "peak_demand_mw":
                annual_run_values[metric] = np.nanmax(arr, axis=1)
            elif metric == "bess_soc_mwh":
                annual_run_values[metric] = arr[:, -1]
            elif metric.startswith("cumulative_"):
                annual_run_values[metric] = arr[:, -1]
            elif metric in {"generation_capacity_mw", "storage_capacity_mwh"}:
                annual_run_values[metric] = arr[:, 0]
            elif metric == "land_use_ha":
                # Installed solar land is sized to the most demanding seasonal
                # coefficient over the simulated period.
                annual_run_values[metric] = np.nanmax(arr, axis=1)
            elif metric == "generation_yield_mwh_per_mw_year":
                annual_run_values[metric] = (
                    np.nansum(daily_results[scenario]["generation_mwh"], axis=1)
                    / np.maximum(1e-9, daily_results[scenario]["generation_capacity_mw"][:, 0])
                    * (365.0 / simulation_days)
                )
            elif metric == "land_use_ha_per_gwh_served":
                annual_run_values[metric] = (
                    np.nanmax(daily_results[scenario]["land_use_ha"], axis=1)
                    / np.maximum(1e-9, np.nansum(
                        daily_results[scenario]["electricity_served_mwh"], axis=1
                    ) / 1000.0)
                )
            elif metric == "land_use_ha_per_gwh_generated":
                annual_run_values[metric] = (
                    np.nanmax(daily_results[scenario]["land_use_ha"], axis=1)
                    / np.maximum(1e-9, np.nansum(
                        daily_results[scenario]["generation_mwh"], axis=1
                    ) / 1000.0)
                )
            elif metric == "sunny_day_share_assumed":
                annual_run_values[metric] = (
                    np.full(mc_runs, np.nan) if np.all(np.isnan(arr)) else np.nanmean(arr, axis=1)
                )
            elif metric == "capacity_factor_realised":
                annual_run_values[metric] = (
                    np.nansum(daily_results[scenario]["generation_mwh"], axis=1)
                    / np.maximum(
                        1e-9,
                        daily_results[scenario]["generation_capacity_mw"][:, 0]
                        * 24.0 * simulation_days,
                    )
                )
            elif metric == "solar_capacity_factor_realised":
                annual_run_values[metric] = (
                    np.full(mc_runs, np.nan) if np.all(np.isnan(arr)) else np.nanmean(arr, axis=1)
                )
            elif metric == "sunny_day_indicator":
                annual_run_values[metric] = (
                    np.full(mc_runs, np.nan) if np.all(np.isnan(arr))
                    else np.nansum(arr, axis=1) / simulation_days
                )
            else:
                annual_run_values[metric] = (
                    np.full(mc_runs, np.nan) if np.all(np.isnan(arr)) else np.nanmean(arr, axis=1)
                )

        annual_period_type = (
            "ANNUAL_SCENARIO" if simulation_days == 365 else "SIMULATION_PERIOD_SCENARIO"
        )
        annual_row = {
            "period_type": annual_period_type,
            "time_period": simulation_days,
            "weather_profile_id": route_weather_profile,
            "weather_season": "mixed" if simulation_days > 10 else "ideal",
            "route_id": route_id,
            "route_name": route_name,
            "country_region": country_region,
            "scenario": scenario,
            "station_id": "",
            "station_name": "",
            "route_length_km": route_length_km,
            "number_of_stations": n_stations,
            "barren_soil_km": barren_soil_km,
            "segment_barren_soil_sum_km": segment_barren_soil_km,
            "barren_soil_share": barren_soil_share,
            "monte_carlo_runs": mc_runs,
        }
        for metric, values in annual_run_values.items():
            if np.all(np.isnan(values)):
                annual_row[metric] = np.nan
                annual_row[f"{metric}_p05"] = np.nan
                annual_row[f"{metric}_p95"] = np.nan
            else:
                annual_row[metric] = q50(values)
                annual_row[f"{metric}_p05"] = q05(values)
                annual_row[f"{metric}_p95"] = q95(values)
        all_output_rows.append(annual_row)
        annual_summary_rows.append(annual_row.copy())

        for sid in station_ids:
            i = station_index[sid]
            station_row = {
                "period_type": (
                    "ANNUAL_STATION" if simulation_days == 365 else "SIMULATION_PERIOD_STATION"
                ),
                "time_period": simulation_days,
                "weather_profile_id": route_weather_profile,
                "weather_season": "mixed" if simulation_days > 10 else "ideal",
                "route_id": route_id,
                "route_name": route_name,
                "country_region": country_region,
                "scenario": scenario,
                "station_id": sid,
                "station_name": station_names[sid],
                "route_length_km": route_length_km,
                "number_of_stations": n_stations,
                "barren_soil_km": barren_soil_km,
                "segment_barren_soil_sum_km": segment_barren_soil_km,
                "barren_soil_share": barren_soil_share,
                "monte_carlo_runs": mc_runs,
            }
            for metric in station_annual_metric_names:
                values = station_annual_results[scenario][metric][:, i]
                station_row[metric] = q50(values)
                station_row[f"{metric}_p05"] = q05(values)
                station_row[f"{metric}_p95"] = q95(values)
            all_output_rows.append(station_row)

# -----------------------------------------------------------------------------
# 6. Add wide three-route comparison rows
# -----------------------------------------------------------------------------
annual_summary = pd.DataFrame(annual_summary_rows)
comparison_metrics = [
    "route_length_km", "number_of_stations", "external_demand_mwh",
    "electricity_served_mwh", "energy_not_served_mwh",
    "electricity_service_ratio", "generation_capacity_mw",
    "storage_capacity_mwh", "generation_mwh", "grid_import_mwh",
    "grid_import_share", "capacity_factor_realised",
    "solar_capacity_factor_realised",
    "sunny_day_share_assumed", "weather_generation_coefficient",
    "weather_price_coefficient", "weather_land_coefficient",
    "grid_price_coefficient", "total_cost_eur",
    "levelized_delivered_electricity_cost_eur_per_mwh",
    "cost_eur_per_passenger_km", "lifecycle_emissions_kgco2e",
    "emissions_gco2e_per_passenger_km", "land_use_ha",
    "generation_yield_mwh_per_mw_year", "land_use_ha_per_gwh_served",
    "land_use_ha_per_gwh_generated", "on_time_departure_ratio",
]
comparison_rows = []
comparison_route_ids = [rid for rid in ROUTE_ORDER if rid in annual_summary["route_id"].values]
if len(comparison_route_ids) >= 2:
    for scenario in SCENARIOS:
        route_records = {}
        for rid in comparison_route_ids:
            match = annual_summary[
                (annual_summary["route_id"] == rid)
                & (annual_summary["scenario"] == scenario)
            ]
            if not match.empty:
                route_records[rid] = match.iloc[0]
        if len(route_records) < 2:
            continue
        row = {
            "period_type": "ROUTE_COMPARISON",
            "time_period": simulation_days,
            "weather_profile_id": weather_profile_id,
            "weather_season": "mixed" if simulation_days > 10 else "ideal",
            "route_id": "_vs_".join(route_records.keys()),
            "route_name": "route_comparison",
            "country_region": "Latvia / California, USA / Maharashtra, India",
            "scenario": scenario,
            "station_id": "",
            "station_name": "",
            "monte_carlo_runs": mc_runs,
        }
        for rid, rec in route_records.items():
            row[f"{rid}_name"] = rec["route_name"]
            row[f"{rid}_country_region"] = rec["country_region"]
            for metric in comparison_metrics:
                row[f"{rid}_{metric}"] = rec.get(metric, np.nan)

        route_pairs = []
        for i, base_rid in enumerate(route_records):
            for other_rid in list(route_records)[i + 1:]:
                route_pairs.append((base_rid, other_rid))
        for base_rid, other_rid in route_pairs:
            base = route_records[base_rid]
            other = route_records[other_rid]
            for metric in comparison_metrics:
                bv = base.get(metric, np.nan)
                ov = other.get(metric, np.nan)
                row[f"{other_rid}_vs_{base_rid}_{metric}_pct"] = (
                    (ov - bv) / bv * 100.0
                    if not pd.isna(bv) and not pd.isna(ov)
                    and not math.isclose(float(bv), 0.0)
                    else np.nan
                )
        comparison_rows.append(row)
        all_output_rows.append(row)

output = pd.DataFrame(all_output_rows)
id_columns = [
    "period_type", "time_period", "weather_profile_id", "weather_season",
    "route_id", "route_name", "country_region",
    "scenario", "station_id", "station_name", "route_length_km",
    "number_of_stations", "barren_soil_km", "segment_barren_soil_sum_km",
    "barren_soil_share", "monte_carlo_runs"
]
existing_id_columns = [c for c in id_columns if c in output.columns]
other_columns = [c for c in output.columns if c not in existing_id_columns]
output = output[existing_id_columns + other_columns]
output.to_csv(OUTPUT_FILE, index=False, float_format="%.6f")

route_comparison = pd.DataFrame(comparison_rows)
route_comparison.to_csv(COMPARISON_FILE, index=False, float_format="%.6f")

annual_display_columns = [
    "weather_profile_id", "route_id", "route_name", "scenario", "route_length_km",
    "capacity_factor_realised", "solar_capacity_factor_realised",
    "weather_generation_coefficient",
    "weather_price_coefficient", "weather_land_coefficient",
    "external_demand_mwh",
    "electricity_served_mwh", "energy_not_served_mwh", "total_cost_eur",
    "levelized_delivered_electricity_cost_eur_per_mwh",
    "cost_eur_per_passenger_km", "lifecycle_emissions_kgco2e", "land_use_ha",
    "generation_yield_mwh_per_mw_year", "land_use_ha_per_gwh_served"
]
annual_table = output[output["period_type"].isin([
    "ANNUAL_SCENARIO", "SIMULATION_PERIOD_SCENARIO"
])][
    [c for c in annual_display_columns if c in output.columns]
].reset_index(drop=True)
annual_table.to_csv(SUMMARY_FILE, index=False, float_format="%.6f")

print(
    f"Simulation completed: {len(route_ids)} routes x {len(SCENARIOS)} scenarios, "
    f"{mc_runs} Monte Carlo runs, {simulation_days} daily periods, "
    f"weather profile '{weather_profile_id}'."
)
print(f"Output saved to: {Path(OUTPUT_FILE).resolve()}")
print(f"Route comparison saved to: {Path(COMPARISON_FILE).resolve()}")
print(f"Annual route-scenario summary saved to: {Path(SUMMARY_FILE).resolve()}")
display(annual_table)
if not route_comparison.empty:
    display(route_comparison)
