"""Lower-limb LimbConfig — SKELETON (fill in from measurements before fitting).

Chain (proximal -> distal):  spine -> pelvis -> hip -> thigh -> shin -> ankle -> foot.
This mirrors limbs/arm.py: write the topology and drop in measured masses / areas /
resistances, then it runs on the SAME engine (core.ThermalROM) — no new solver code.

AREA is REAL (measured external heat-transfer area, right side; total 0.5177 m^2).
Everything else tagged `# TODO` is still a placeholder.

================================ TODО / FILL IN ================================
  * ACTS / TOPO   — confirm the real joint set & DOF (hip is typically 3-DOF:
                    yaw/roll/pitch; ankle 2-DOF). The serial 1-actuator-per-joint
                    chain below is a placeholder so the config constructs.
  * C_M / C_WIND  — motor assembly + winding C [J/K] (Node 1 tables), one per actuator.
  * C_S_FIX       — structure thermal masses [J/K] (Node 2), per segment.
  * FABRICS / C_F_FIX / R_STACK — which segments wear fabric + fabric mass / stack R.
  * TS_COLS       — leg thermocouple CSV column names.
  * BOUNDARY_ACTS / ENCLOSED — driven boundary joint; any no-ambient segment.
  * EXTRA_Q       — any constant electronics load.
  * TRAIN / TEST  — data tags once data/leg/data_<tag>.csv exist.
===============================================================================
"""
import numpy as np
from core import LimbConfig

# 8 measured segments (the heat-transfer-area node set)
_STRUCTS = ['spine', 'pelvis', 'hip', 'thigh_u', 'thigh_l', 'shin', 'ankle', 'foot']

# Measured external heat-transfer area per segment [m^2] (right side). mm^2 in comment.
_AREA = {
    'spine':   0.0791,   #  79,115 mm^2
    'pelvis':  0.0575,   #  57,502 mm^2
    'hip':     0.0273,   #  27,292 mm^2
    'thigh_u': 0.1622,   # 162,178 mm^2  (upper thigh)
    'thigh_l': 0.0383,   #  38,262 mm^2  (lower thigh / knee)
    'shin':    0.1322,   # 132,216 mm^2  (lower leg / shin)
    'ankle':   0.0093,   #   9,280 mm^2
    'foot':    0.0119,   #  11,874 mm^2
}                        # total = 0.5177 m^2

# Placeholder serial chain: one actuator per inter-segment joint, plus the spine boundary.
# 8 segments -> 8 actuators (1 boundary + 7 links), mirroring the arm's J1+6-link layout. TODO real DOF.
_ACTS = ['SPINE', 'HIP_YAW', 'HIP_ROLL', 'HIP_PITCH', 'KNEE', 'ANKLE_PITCH', 'ANKLE_ROLL', 'FOOT']
_N = len(_ACTS)

CONFIG = LimbConfig(
    name='leg',
    ACTS=_ACTS,
    STRUCTS=_STRUCTS,
    FABRICS=['thigh_u', 'thigh_l', 'shin'],          # TODO confirm covered segments
    TOPO={                                            # TODO confirm (housing, output) per joint
        'SPINE':       ('spine', 'spine'),            # boundary: housing==output (no link)
        'HIP_YAW':     ('spine', 'pelvis'),           # link spine -> pelvis
        'HIP_ROLL':    ('pelvis', 'hip'),
        'HIP_PITCH':   ('hip', 'thigh_u'),
        'KNEE':        ('thigh_u', 'thigh_l'),
        'ANKLE_PITCH': ('thigh_l', 'shin'),
        'ANKLE_ROLL':  ('shin', 'ankle'),
        'FOOT':        ('ankle', 'foot'),
    },
    TS_COLS={                                         # TODO match the leg thermocouple CSV columns
        'spine': 'Ts_spine', 'pelvis': 'Ts_pelvis', 'hip': 'Ts_hip',
        'thigh_u': 'Ts_thigh_u', 'thigh_l': 'Ts_thigh_l', 'shin': 'Ts_shin',
        'ankle': 'Ts_ankle', 'foot': 'Ts_foot',
    },

    C_M=np.full(_N, 150.0),                           # TODO motor assembly C [J/K]
    C_WIND=np.full(_N, 15.0),                         # TODO winding C [J/K]
    C_S_FIX={s: 400.0 for s in _STRUCTS},             # TODO structure C [J/K]
    C_F_FIX={'thigh_u': 40.0, 'thigh_l': 40.0, 'shin': 30.0},   # TODO fabric C [J/K]

    R_STACK={'thigh_u': 1.0, 'thigh_l': 1.0, 'shin': 1.0},      # TODO structure->fabric [K/W]
    AREA=_AREA,                                       # measured (see above)

    QFET=4.0,                                         # TODO confirm FET load
    EXTRA_Q={},                                       # TODO any constant electronics load

    BOUNDARY_ACTS=['SPINE'],                          # TODO the measured spine/torso-side boundary
    ENCLOSED={},                                      # every segment has a measured ambient area

    DPDT_THRESH=5.0, CADAPT_TAU=80.0, ADAPT_C=True, C_WIND_SCALE=2.5,

    TRAIN=[],                                         # TODO data tags once collected
    TEST=[],
)
