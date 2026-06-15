# 1D Thermal ROM

A reduced-order, lumped-parameter (RC-network) thermal model of a serial actuator
limb. This is the **1D** companion to the 2D web model at the repo root — they share
the repo but are independent: this folder has no Flask/web dependency.

Given a measured power/temperature run it predicts every motor and structure
temperature over time, and supports energy audits and continuous-current-limit
back-calculation.

## The one idea: engine + per-limb config

All the math is **limb-agnostic** and lives in `core/`. A limb (arm, leg, …) is
**just a configuration object** — node lists, chain topology, measured masses,
areas, resistances, and data-column names. Adding a limb means writing one config,
not copying the solver.

```
rom_1d/
├── core/
│   ├── engine.py        # ThermalROM: network assembly, ZOH discretization,
│   │                    #   adaptive motor C, simulate, energy, residuals
│   └── limb_config.py   # LimbConfig dataclass (the only thing that varies per limb)
├── limbs/
│   ├── arm.py           # left-arm config (J1..YAW)  — fully populated
│   └── leg.py           # spine→pelvis→hip→thigh→shin→foot — SKELETON w/ TODOs
├── pipeline/            # limb-parametrized tools (take --limb)
│   ├── fit.py           # grey-box least-squares fit of R2, R_link, b
│   ├── plot_fit.py      # train/test overview (--split train|test)
│   ├── plot_hi19.py     # arm 3-joint endurance validation
│   ├── energy.py        # balance / distribution / time-resolved audit (--mode)
│   └── current_limit.py # steady-state continuous current limit per actuator
├── assemble/
│   └── assemble_arm.py  # build data/arm/data_*.csv from raw test logs
├── data/<limb>/         # assembled data_<tag>.csv
├── fits/<limb>_fitted_params.json
├── figures/             # generated PNGs (gitignored)
└── main.py              # python main.py --limb arm [steps|all]
```

## Model

18 states for the arm: 7 motor + 7 structure + 4 fabric nodes.
- `motor_i --R2_i--> housing structure (metal)`; structures bridged through the
  actuator by `R_link_i` (collapsed gearbox path).
- Covered structures: `structure --R_stack--> fabric node --1/(b·A)--> air` (TC on fabric).
- Bare structures: direct `1/(b·A)` to air (TC on metal).
- Enclosed structures (e.g. the shoulder in the torso shell): no ambient path.
- Boundary actuators (e.g. J1): clamped to their measured temperature each step.
- FET/driver heat (always on) into each housing; optional constant electronics loads.
- **Adaptive motor capacitance**: a sudden `|dP/dt|` transient dips the motor heat
  mass to the winding-only value; between transients it recovers toward the full
  assembly mass with time constant τ. Steady or slowly-ramping power → full C.

Only `R2`, `R_link`, and the per-surface convective `b` are fit (log10-encoded,
`scipy.least_squares`); all capacitances are fixed from measured masses.

## Usage

```bash
conda activate thermal_sim          # numpy / scipy / pandas / matplotlib

# arm: fit + train/test plots
python main.py --limb arm

# everything (adds hi19 validation, energy audit, current limit)
python main.py --limb arm all

# individual tools
python pipeline/current_limit.py --limb arm --tamb 25
python pipeline/energy.py --limb arm --mode dist
```

## Adding the lower limb

1. Fill in `limbs/leg.py` (every `# TODO`: topology, masses, areas, resistances,
   boundary/enclosed nodes). Register it in `limbs/__init__.py` (already listed).
2. Write `assemble/assemble_leg.py` to produce `data/leg/data_<tag>.csv` from the
   leg test logs (mirror `assemble_arm.py`).
3. `python main.py --limb leg fit trainplot` — same engine, same pipeline.
