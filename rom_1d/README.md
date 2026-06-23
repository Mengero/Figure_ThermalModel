# 1D Thermal ROM

A reduced-order, lumped-parameter (RC-network) thermal model of a serial actuator
limb. This is the **1D** companion to the 2D web model at the repo root — they share
the repo but are independent: this folder has no Flask/web dependency.

Given a measured power/temperature run it predicts every motor and structure
temperature over time, and supports energy audits, continuous-current-limit
back-calculation, sensor-fault detection, and a single-file deployable export.

## The one idea: engine + per-limb config

All the math is **limb-agnostic** and lives in `core/`. A limb (arm, leg, …) is
**just a configuration object** — node lists, chain topology, measured masses,
areas, resistances, and data-column names. Adding a limb means writing one config,
not copying the solver.

```
rom_1d/
├── core/                 # the model (limb-agnostic)
│   ├── engine.py           ThermalROM: network assembly, ZOH discretization,
│   │                         adaptive motor C, simulate, energy, residuals
│   ├── limb_config.py       LimbConfig dataclass (the only thing that varies per limb)
│   └── observer.py          ThermalObserver: fault detection + virtual sensor (FDI)
├── limbs/
│   ├── arm.py               left-arm config (J1..YAW) — fully populated
│   └── leg.py               spine→…→foot — SKELETON w/ TODOs
├── deploy/
│   └── arm_thermal_rom.py    single self-contained fitted arm model (share this file)
├── pipeline/             # the fit/validate flow driven by main.py + shared helpers
│   ├── _common.py           paths, rom(), load(), load_params()
│   ├── fields.py            load_field(): robot long-format CSV -> per-step DataFrame
│   ├── plots.py             shared per-motor grid plot helpers
│   ├── _faults.py           synthetic fault injection
│   ├── fit.py  plot_fit.py  plot_hi19.py  test_cont.py  test_steady.py
│   └── validate_fdi.py  energy.py  current_limit.py
├── studies/              # by-hand analysis (NOT part of main.py)
│   ├── dropout.py           sensor-dropout accuracy (--mode midrun|boot|solo|allzero)
│   ├── field_accuracy.py    open-loop experiment-vs-sim accuracy on robot data
│   └── plot_capacitance.py  adaptive motor C(t) vs power
├── assemble/
│   └── assemble_arm.py       build data/arm/data_*.csv from raw test logs
├── data/<limb>/          assembled data_<tag>.csv  (+ data/ups, data/depal field CSVs)
├── fits/<limb>_fitted_params.json
├── figures/              generated PNGs (gitignored)
└── main.py               python main.py --limb arm [steps|all]
```

## Model

18 states for the arm: 7 motor + 7 structure + 4 fabric nodes.
- `motor_i --R2_i--> housing structure (metal)`; structures bridged through the
  actuator by `R_link_i` (collapsed gearbox path).
- Covered structures: `structure --R_stack--> fabric node --1/(b·A)--> air` (TC on fabric).
- Bare structures (wrist, hand): direct `1/(b·A)` to air.
- Enclosed structures (shoulder in the torso shell): no ambient path.
- Torso boundary: the J1 motor sinks to the battery/torso temp through a fitted `R_torso`.
- FET/driver heat (always on) into each housing; optional constant electronics loads.
- **Adaptive motor capacitance**: a sudden `|dP/dt|` transient dips the motor heat
  mass to the winding-only value; between transients it recovers toward the full
  assembly mass with time constant τ.

Only `R2`, `R_link`, the per-surface convective `b`, and `R_torso` are fit
(log10-encoded, `scipy.least_squares`); all capacitances are fixed from measured masses.

Copper power convention: logs store `power_W = ¾·iq²`; the real copper loss used
everywhere is `1.5·iq²·R(T) = 2·power_W·R20·(234.5+T)/254.5`.

--------------------------------------------------------------------------------

## HOW TO RUN — command reference

Always activate the env first:

```bash
conda activate thermal_sim          # numpy / scipy / pandas / matplotlib
cd rom_1d
```

### The pipeline (fit + standard validation) — `main.py`

```bash
python main.py --limb arm                 # fit + train plot + test plot (default steps)
python main.py --limb arm all             # every step below, in order
python main.py --limb arm fit testplot    # pick specific steps
```
Steps: `fit  trainplot  testplot  hi19  conttest  fdi  energy  climit`.

### Individual pipeline tools (and the figure each writes)

| What you want | Command | Figure → `figures/` |
|---|---|---|
| Re-fit the parameters | `python pipeline/fit.py --limb arm` | *(writes `fits/arm_fitted_params.json`)* |
| Training-set overview | `python pipeline/plot_fit.py --limb arm --split train` | `arm_train_fit.png` |
| Test-set overview | `python pipeline/plot_fit.py --limb arm --split test` | `arm_test_fit.png` |
| hi19 endurance validation | `python pipeline/plot_hi19.py --limb arm` | `arm_hi19_validation.png` |
| Continuous-run held-out test | `python pipeline/test_cont.py --limb arm --case 0616cont` | `arm_0616cont_test.png` |
| Steady-state held-out test | `python pipeline/test_steady.py --limb arm` | *(prints table)* |
| Energy audit | `python pipeline/energy.py --limb arm --mode all` | `arm_energy_dist.png`, `arm_energy_time.png` |
| Continuous current limit | `python pipeline/current_limit.py --limb arm --tamb 25` | `arm_current_limit_transient.png` |

### Fault detection / virtual sensor — `pipeline/validate_fdi.py`

```bash
# inject a synthetic fault into a clean case (detection latency + virtual-sensor error)
python pipeline/validate_fdi.py --case 0616cont --dt 2 --inject YAW dead 60 120
#                                                          ^sensor ^type ^t0 ^t1  (type: dead|nan|oor|stuck)
python pipeline/validate_fdi.py --real            # run on c_1076 (real dead YAW)
python pipeline/validate_fdi.py --sweep 6 --dt 2  # k=1..6 simultaneous dropouts -> usability limit
```
Figures: `arm_fdi_<case>_<sensor>_<type>.png`, `arm_fdi_real_*.png`, `arm_fdi_sweep_*.png`.

### Analysis studies — `studies/` (run by hand)

**Sensor-dropout accuracy** (virtual prediction vs withheld truth):
```bash
# one joint drops mid-run, just after a heavy lift (sweep all 7)
python studies/dropout.py --mode midrun  --case data/depal/c_1092_depal_cleaned.csv --dt 4 --t0 82
# one joint dead from boot (sweep all 7)
python studies/dropout.py --mode boot    --case data/depal/c_1092_depal_cleaned.csv --dt 4
# only ONE thermistor survives, six drop -> RMSE heatmap (+ --survivor J2 for its time series)
python studies/dropout.py --mode solo    --case data/depal/c_1092_depal_cleaned.csv --dt 4 --survivor J2
# ALL sensors dead, model runs open-loop seeded from the battery temp
python studies/dropout.py --mode allzero --case data/depal/c_1092_depal_cleaned.csv --dt 4
```
Figures: `dropout_<mode>_<case>[...].png`. Use any field CSV under `data/ups/` or
`data/depal/` (use `--dt 10` for the non-`*_cleaned` long-format files).

**Open-loop field accuracy** (experiment vs simulation):
```bash
python studies/field_accuracy.py --all                                         # table + figures/field_compare.png
python studies/field_accuracy.py --csv data/ups/c_1081_ups_cleaned.csv --dt 4  # one case -> figures/<case>_predict.png
```

**Adaptive motor capacitance** C(t):
```bash
python studies/plot_capacitance.py                                           # depal + ups -> figures/capacitance_vs_time.png
python studies/plot_capacitance.py --cases data/depal/c_1092_depal_cleaned.csv --tmax 10   # zoom first 10 min
```

### Regenerate ALL figures from scratch

```bash
python main.py --limb arm all                                   # all pipeline figures
python studies/field_accuracy.py --all
for c in data/ups/c_1081_ups_cleaned.csv data/ups/c_1261_ups_cleaned.csv \
         data/depal/c_1092_depal_cleaned.csv data/depal/c_1076_depal.csv; do
  python studies/field_accuracy.py --csv "$c" --dt 4
done
python studies/plot_capacitance.py
for m in midrun boot solo allzero; do
  python studies/dropout.py --mode $m --case data/depal/c_1092_depal_cleaned.csv --dt 4
done
```

### Deployable single-file model

`deploy/arm_thermal_rom.py` is self-contained (numpy only; scipy optional) with the
fitted parameters baked in — share **just this one file**. Demo:
```bash
python deploy/arm_thermal_rom.py
```
Online use: `rom = ArmThermalROM(); x = rom.init_state(...); x = rom.step(x, iq, Tamb, Tbatt, dt)`.

--------------------------------------------------------------------------------

## Adding the lower limb

1. Fill in `limbs/leg.py` (every `# TODO`: topology, masses, areas, resistances,
   boundary/enclosed nodes). It's already registered in `limbs/__init__.py`.
2. Write `assemble/assemble_leg.py` to produce `data/leg/data_<tag>.csv` from the
   leg test logs (mirror `assemble_arm.py`).
3. `python main.py --limb leg fit trainplot` — same engine, same pipeline.
