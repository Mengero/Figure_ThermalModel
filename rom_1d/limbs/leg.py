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

# Spine has TWO SM85 motors (SPN Z + SPN X). Distal joints are placeholder serial links
# (one actuator per inter-segment joint). 8 segments, 8 motors. TODO real DOF below the pelvis.
_ACTS = ['SPN_Z', 'SPN_X', 'HIP_YAW', 'HIP_ROLL', 'HIP_PITCH', 'KNEE', 'ANKLE_PITCH', 'ANKLE_ROLL']
_N = len(_ACTS)

# --- spine thermal masses (MEASURED, SM85; halved for left/right symmetry) ---
#   Node 1a/1b stator+rotor: 237.5 J/K each -> /2 = 118.75 ;  windings 74.8 -> /2 = 37.4
#   Node 2 structural+drivetrain+electronics: 1040.7 J/K -> /2 = 520.35
_SPN_CM = 237.5 / 2     # 118.75 J/K per spine motor (assembly)
_SPN_CW = 74.8 / 2      # 37.40  J/K per spine motor (windings)
_SPN_CS = 1040.7 / 2    # 520.35 J/K spine structure (Node 2)

CONFIG = LimbConfig(
    name='leg',
    ACTS=_ACTS,
    STRUCTS=_STRUCTS,
    FABRICS=['thigh_u', 'thigh_l', 'shin'],          # TODO confirm covered segments
    TOPO={                                            # TODO confirm (housing, output) per joint below pelvis
        'SPN_Z':       ('spine', 'spine'),            # spine Z motor: leaf on spine (no link)
        'SPN_X':       ('spine', 'pelvis'),           # spine X motor: links spine -> pelvis
        'HIP_YAW':     ('pelvis', 'hip'),
        'HIP_ROLL':    ('hip', 'thigh_u'),
        'HIP_PITCH':   ('thigh_u', 'thigh_l'),
        'KNEE':        ('thigh_l', 'shin'),
        'ANKLE_PITCH': ('shin', 'ankle'),
        'ANKLE_ROLL':  ('ankle', 'foot'),
    },
    TS_COLS={                                         # TODO match the leg thermocouple CSV columns
        'spine': 'Ts_spine', 'pelvis': 'Ts_pelvis', 'hip': 'Ts_hip',
        'thigh_u': 'Ts_thigh_u', 'thigh_l': 'Ts_thigh_l', 'shin': 'Ts_shin',
        'ankle': 'Ts_ankle', 'foot': 'Ts_foot',
    },

    # motor assembly C [J/K]: SPN_Z, SPN_X measured (SM85/2); the rest TODO
    C_M=np.array([_SPN_CM, _SPN_CM, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0]),
    # winding C [J/K]: SPN_Z, SPN_X measured (SM85/2); the rest TODO
    C_WIND=np.array([_SPN_CW, _SPN_CW, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0]),
    C_S_FIX={'spine': _SPN_CS, 'pelvis': 400.0, 'hip': 400.0, 'thigh_u': 400.0,
             'thigh_l': 400.0, 'shin': 400.0, 'ankle': 400.0, 'foot': 400.0},   # spine measured; rest TODO
    C_F_FIX={'thigh_u': 40.0, 'thigh_l': 40.0, 'shin': 30.0},   # TODO fabric C [J/K]

    R_STACK={'thigh_u': 1.0, 'thigh_l': 1.0, 'shin': 1.0},      # TODO structure->fabric [K/W]
    AREA=_AREA,                                       # measured (see above)

    QFET=4.0,                                         # TODO confirm FET load
    EXTRA_Q={},                                       # TODO any constant electronics load

    BOUNDARY_ACTS=[],                                 # TODO no measured torso boundary yet; spine motors driven
    ENCLOSED={},                                      # every segment has a measured ambient area

    # leg uses the MEASURED winding mass directly (scale 1.0); the arm used 2.5 as a fit fudge.
    DPDT_THRESH=5.0, CADAPT_TAU=80.0, ADAPT_C=True, C_WIND_SCALE=1.0,

    TRAIN=[],                                         # TODO data tags once collected
    TEST=[],
)
