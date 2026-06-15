"""Lower-limb LimbConfig — SKELETON (fill in from measurements before fitting).

Chain (proximal -> distal):  spine -> pelvis -> hip -> thigh -> shin -> foot.
This mirrors limbs/arm.py: write the topology and drop in measured masses / areas /
resistances, then it runs on the SAME engine (core.ThermalROM) — no new solver code.

================================ TODО / FILL IN ================================
Every value tagged `# TODO` is a placeholder copied loosely from the arm so the
config imports and the shapes line up. Replace each with the real F03 lower-limb
number before trusting any result:
  * ACTS / TOPO   — confirm the actual joint set and which structure each motor's
                    housing and output attach to (hip is typically 3 DOF: yaw/roll/pitch).
  * C_M / C_WIND  — motor assembly + winding masses*specific-heat (Node 1 tables).
  * C_S_FIX       — structure thermal masses (Node 2).
  * C_F_FIX / R_STACK — fabric mass and structure->fabric resistance for covered segments.
  * AREA          — externally-exposed area per segment [m^2].
  * BOUNDARY_ACTS — the driven boundary joint (the spine/torso-side analog of the arm's J1).
  * ENCLOSED      — any segment with no ambient path (inherits a neighbour's metal temp).
  * EXTRA_Q       — any constant electronics load (analog of the arm's hand LV).
  * TRAIN / TEST  — data tags once you have data/leg/data_<tag>.csv.
===============================================================================
"""
import numpy as np
from core import LimbConfig

# Placeholder joint set — ONE actuator per inter-segment joint. Adjust to the real DOF count.
_ACTS = ['SPINE', 'HIP_YAW', 'HIP_ROLL', 'HIP_PITCH', 'KNEE', 'ANKLE']
_STRUCTS = ['pelvis', 'hip', 'thigh_u', 'thigh_l', 'shin', 'foot']
_N = len(_ACTS)

CONFIG = LimbConfig(
    name='leg',
    ACTS=_ACTS,
    STRUCTS=_STRUCTS,
    FABRICS=['thigh_u', 'thigh_l', 'shin'],          # TODO confirm which segments wear fabric
    TOPO={                                            # TODO (housing, output) per joint
        'SPINE':     ('pelvis', 'pelvis'),            # boundary joint: housing==output (no link)
        'HIP_YAW':   ('pelvis', 'hip'),
        'HIP_ROLL':  ('hip', 'thigh_u'),
        'HIP_PITCH': ('thigh_u', 'thigh_l'),
        'KNEE':      ('thigh_l', 'shin'),
        'ANKLE':     ('shin', 'foot'),
    },
    TS_COLS={                                         # TODO match the leg thermocouple CSV columns
        'thigh_u': 'Ts_thigh_u', 'thigh_l': 'Ts_thigh_l',
        'shin': 'Ts_shin', 'foot': 'Ts_foot',
    },

    C_M=np.full(_N, 150.0),                           # TODO motor assembly C [J/K]
    C_WIND=np.full(_N, 15.0),                         # TODO winding C [J/K]
    C_S_FIX={s: 400.0 for s in _STRUCTS},             # TODO structure C [J/K]
    C_F_FIX={'thigh_u': 40.0, 'thigh_l': 40.0, 'shin': 30.0},   # TODO fabric C [J/K]

    R_STACK={'thigh_u': 1.0, 'thigh_l': 1.0, 'shin': 1.0},      # TODO structure->fabric [K/W]
    AREA={'pelvis': 0.02, 'hip': 0.02, 'thigh_u': 0.04,         # TODO exposed area [m^2]
          'thigh_l': 0.04, 'shin': 0.03, 'foot': 0.015},

    QFET=4.0,                                         # TODO confirm FET load
    EXTRA_Q={},                                       # TODO any constant electronics load

    BOUNDARY_ACTS=['SPINE'],                          # TODO the measured spine/torso-side boundary
    ENCLOSED={'pelvis': 'thigh_u'},                   # TODO any enclosed (no-ambient) segment

    DPDT_THRESH=5.0, CADAPT_TAU=80.0, ADAPT_C=True, C_WIND_SCALE=2.5,

    TRAIN=[],                                         # TODO data tags once collected
    TEST=[],
)
