# Hyperloop Power Supply Comparison

System-dynamics simulation and comparative assessment of **solar photovoltaic, nuclear, and prospective fusion power supply for battery-electric Hyperloop systems**.

The project evaluates how the three energy-supply scenarios perform over a daily simulation horizon, typically **365 days**, under different geographical and seasonal conditions. The model links Hyperloop operations, electricity demand, battery charging and swapping, grid exchange, generation availability, cost, lifecycle emissions, and land use.

## Research objective

The study compares the technical, economic, environmental, and operational performance of:

- solar photovoltaic generation with stationary storage and grid balancing;
- nuclear electricity supply;
- prospective fusion electricity supply.

System dynamics is used as the modelling approach because it captures how electricity generation, demand, battery-service capacity, weather, and operational performance evolve over time.

## Route scenarios

The current model compares three route configurations:

- **route_a** — Latvia;
- **route_b** — San Francisco–Los Angeles, California, USA;
- **route_c** — Mumbai–Pune, India.

Route-specific inputs include route length, station locations, electricity prices, solar conditions, land requirements, weather coefficients, and other regional assumptions.

## Main model components

The simulation includes:

- Hyperloop passenger demand and pod scheduling;
- traction and fixed-infrastructure electricity demand;
- removable traction-battery packs;
- direct charging and battery swapping;
- charged and depleted battery-pack inventories;
- battery-service station constraints;
- solar, nuclear, and fusion generation scenarios;
- stationary battery storage for the solar scenario;
- grid imports and exports;
- weather and seasonal effects;
- lifecycle emissions and land requirements;
- financial assessment in EUR;
- Monte Carlo uncertainty analysis using P05, median, and P95 outputs.

## Repository workflow

A typical workflow is:

1. Select an input dataset and rename it to `input.csv`.
2. Run the system-dynamics simulation.
3. Generate the assessment tables.
4. Generate normalised comparison figures and dynamic plots.

Example:

```bash
python hyperloop_system_dynamics.py
python result_assessment_ratio.py
python normalised_visualization.py
python system_dynamics_model.py
```

The scripts are also designed to run in Jupyter Notebook or IPython.

## Input data

The simulation reads a single `input.csv` file containing several record types:

- `SEGMENT` — route links and distances;
- `PARAMETER` — global, route, scenario, and station parameters;
- `WEATHER` — daily route-specific weather and seasonal coefficients.

Two common input configurations are used:

- a **365-day seasonal dataset**;
- a **10-day ideal-weather dataset** for model testing.

The selected file should be renamed to:

```text
input.csv
```

## Main outputs

The core simulation produces:

```text
output.csv
route_comparison.csv
annual_route_scenario_summary.csv
```

Additional assessment scripts can produce:

```text
result_assessment.csv
results_assessment_full.csv
comparative_analysis_solar_vs_nuclear.csv
comparative_analysis_solar_vs_fusion.csv
```

Visualisation scripts can produce:

```text
normalised_heatmap.png
bubble_scatter.png
system_dynamics_model.png
system_dynamics_model_full_scale.png
system_dynamics_model_rolling_average.png
system_dynamics_model_cumulative_advantage.png
```

## Key indicators

The model reports indicators including:

- electricity generated and served;
- energy not served;
- grid-import share;
- realised capacity factor;
- charged-pack availability;
- blocked battery-service events;
- on-time departure ratio;
- total simulation-period cost;
- levelised delivered electricity cost;
- cost per passenger-kilometre;
- lifecycle greenhouse-gas emissions;
- additional power-system land requirement;
- land use per GWh served.

## Normalised dynamic assessment

Dynamic scenario performance is transformed to a common 0–1 scale, where:

- `1` represents the most favourable observed performance;
- `0` represents the least favourable observed performance.

Higher-is-better indicators are normalised directly, while lower-is-better indicators such as cost, emissions, grid imports, and energy not served are reverse-normalised. The resulting trajectories are reported as:

- `solar_avg`;
- `nuclear_avg`;
- `fusion_avg`.

Monte Carlo P05–P95 envelopes are shown around the median dynamic trajectories.

## Comparative interpretation

Solar power is particularly sensitive to geographical and seasonal conditions. In the current scenarios, California provides the strongest solar case, while the Latvian route has lower solar productivity and greater seasonal exposure.

Nuclear power provides comparatively stable generation and low land requirements across routes. Prospective fusion may provide further long-term cost and land-use advantages under mature-technology assumptions.

## Important limitations

- Fusion power is a **prospective scenario** and does not represent an operational commercial technology.
- Fusion cost, availability, recirculating-power demand, and lifetime assumptions are provisional.
- Several Hyperloop parameters are generalised into representative **TRL 6** and **TRL 9** ranges using heterogeneous public project specifications.
- The current main simulation uses a daily time step; shorter time steps may be required for detailed operational and peak-power analysis.
- Results should be interpreted as scenario-based comparative estimates rather than investment-grade forecasts.

## Software requirements

Recommended environment:

- Python 3.10 or newer;
- pandas;
- NumPy;
- Matplotlib;
- Jupyter Notebook or JupyterLab, if using notebooks.

Install the main dependencies with:

```bash
pip install pandas numpy matplotlib jupyter
```

## Research and policy relevance

The project supports research on low-carbon and digitally enabled transport infrastructure. It is relevant to:

- the European Green Deal;
- the EU Sustainable and Smart Mobility Strategy;
- digital transformation of transport and infrastructure planning;
- SDG 7 — Affordable and Clean Energy;
- SDG 9 — Industry, Innovation and Infrastructure;
- SDG 11 — Sustainable Cities and Communities;
- SDG 12 — Responsible Consumption and Production;
- SDG 13 — Climate Action.

## Citation

This repository accompanies ongoing research on comparative power-supply assessment for Hyperloop transportation systems. A complete citation will be added after publication.

## Author

**Aleksejs Vesjolijs**  
Transport and Telecommunication Institute, Riga, Latvia
