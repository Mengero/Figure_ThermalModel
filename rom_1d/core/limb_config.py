"""LimbConfig — everything that makes a limb a *specific* limb.

The thermal-network math lives in engine.ThermalROM and is limb-agnostic; this
object is the only thing that changes between the arm, the lower limb, etc.  Add
a new limb by writing one of these (see limbs/arm.py for a worked example and
limbs/leg.py for a skeleton) — never by copying the engine.

Topology convention (a serial actuator chain):
  motor_i --R2_i--> housing structure (metal).  Structures are bridged THROUGH
  the actuator by R_link_i (the collapsed gearbox path) from housing to output.
  Covered structures wear fabric: structure --R_stack--> fabric node --1/(b*A)--> air,
  TC on the fabric.  Bare structures go straight to air (1/(b*A)), TC on the metal.
  Enclosed structures have NO ambient path and inherit their initial temp from a
  neighbour.  Boundary actuators are clamped to a measured temperature every step
  (a driven boundary, not a prediction target).
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class LimbConfig:
    name: str

    # --- topology ---
    ACTS:    list            # actuators, in chain order  e.g. ['J1','J2',...]
    STRUCTS: list            # structure nodes, in chain order
    FABRICS: list            # subset of STRUCTS that are fabric-covered
    TOPO:    dict            # act -> (housing_struct, output_struct)
    TS_COLS: dict            # measured struct -> CSV column name

    # --- thermal masses [J/K] (fixed, from measured part masses) ---
    C_M:     np.ndarray      # motor full-assembly C, one per actuator
    C_WIND:  np.ndarray      # motor winding-only C, one per actuator (adaptive-C floor)
    C_S_FIX: dict            # struct -> C
    C_F_FIX: dict            # fabric -> C

    # --- conduction / geometry ---
    R_STACK: dict            # covered struct -> structure->fabric resistance [K/W]
    AREA:    dict            # struct -> externally-exposed area [m^2]

    # --- heat inputs ---
    QFET:    float = 4.0     # FET/driver heat into each housing, always on (energized)
    EXTRA_Q: dict = field(default_factory=dict)   # act -> extra constant load [W] (e.g. hand LV)

    # --- special nodes ---
    BOUNDARY_ACTS: list = field(default_factory=list)   # clamped to measured Tm each step
    ENCLOSED: dict = field(default_factory=dict)        # struct -> neighbour it inherits init temp from
    TORSO: tuple = None                                 # (struct, temp[C]): fixed-T sink via fitted R_torso

    # --- adaptive motor capacitance (rate-triggered dip + exponential recovery) ---
    DPDT_THRESH: float = 5.0     # |dP/dt| [W/s] that fully dips C to winding
    CADAPT_TAU:  float = 40.0    # recovery time constant [s] back toward full C
    ADAPT_C:     bool  = True    # False -> constant full C_M
    C_WIND_SCALE: float = 2.5    # tuning multiplier applied to C_WIND

    # --- electrical (optional; used by pipeline/current_limit.py) ---
    R20:   dict = field(default_factory=dict)   # actuator -> phase resistance at 20 C [ohm]
    MOTOR: dict = field(default_factory=dict)   # actuator -> motor family label (display only)
    FIX_R2: bool = False                        # fit: pin R2 to size-based values (fit.py honors this; no --fix-r2 flag needed)
    TIE_R2: bool = False                        # fit: tie R2 across motors of the same size (one fitted value per size, not pinned)
    HALVED_ACTS: tuple = ()                      # motors modeled at half (centerline/shared): C, power, area all halved -> R2 x2 in fit

    # --- train/test split (tags of data/<limb>/data_<tag>.csv) ---
    TRAIN: list = field(default_factory=list)
    TEST:  list = field(default_factory=list)
    STEADY_CASE: str = ''                        # steady-state operating point (data/<limb>/<case>.json) folded into fit; '' = none

    def __post_init__(self):
        self.C_M = np.asarray(self.C_M, float)
        self.C_WIND = np.asarray(self.C_WIND, float) * self.C_WIND_SCALE
        # boundary actuators have no winding<->assembly transition (C constant)
        for a in self.BOUNDARY_ACTS:
            self.C_WIND[self.ACTS.index(a)] = self.C_M[self.ACTS.index(a)]
