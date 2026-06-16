"""Lower-limb LimbConfig — masses/areas loaded from measured part tables.

Chain (proximal -> distal):  spine -> pelvis -> hip -> thigh_u -> thigh_l -> shin
-> ankle(talus) -> foot.  Runs on the SAME engine (core.ThermalROM) as the arm.

Motor inventory (8 motors), by the segment each is housed in:
  spine   : SPN_Z, SPN_X        (2x SM85)        — halved (shared/symmetric)
  pelvis  : HIP_Y               (SM85)           — halved (shared/symmetric)
  hip     : HIP_X               (SM72)           — exact
  thigh_u : THIGH_U             (SM72)           — exact
  thigh_l : (none — passive segment)
  shin    : SHIN_KNEE (SM72), SHIN_ANKLE (SM85)  — exact
  ankle   : TALUS               (SM44L)          — exact
Covered (fabric) segments: hip, thigh_u, thigh_l, shin (each has a Node 3 covering).
Bare segments: spine, pelvis, ankle, foot.

AREA is measured (right side, total 0.5177 m^2). Spine & pelvis are halved per the
left/right symmetry instruction; hip and below are loaded exact.

================================ TODО / FILL IN ================================
  * C_S_FIX['foot'] — no foot table yet (placeholder); confirm if foot has a motor.
  * TS_COLS — actual leg thermocouple CSV column names.
  * R20 / MOTOR — per-actuator phase resistance for current_limit.py.
  * TRAIN / TEST — data tags once data/leg/data_<tag>.csv exist.
  * Confirm joint->segment housing map (esp. that shin houses both knee & ankle motors).
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
    'ankle':   0.0093,   #   9,280 mm^2  (talus)
    'foot':    0.0119,   #  11,874 mm^2
}                        # total = 0.5177 m^2

# 8 real motors, ordered by housing segment (proximal -> distal)
_ACTS = ['SPN_Z', 'SPN_X', 'HIP_Y', 'HIP_X', 'THIGH_U', 'SHIN_KNEE', 'SHIN_ANKLE', 'TALUS']

# ---- motor thermal masses [J/K]  (assembly = Node 1 ; winding = Winding TM) ----
# spine + pelvis SM85 are HALVED (shared/symmetric); hip and below are EXACT.
_CM = {  # assembly thermal mass
    'SPN_Z': 237.5 / 2, 'SPN_X': 237.5 / 2,   # SM85  (halved) = 118.75
    'HIP_Y': 237.5 / 2,                        # SM85  (halved) = 118.75
    'HIP_X': 151.0, 'THIGH_U': 151.0, 'SHIN_KNEE': 151.0,   # SM72  (exact)
    'SHIN_ANKLE': 237.5,                       # SM85  (exact)
    'TALUS': 259.9,                            # SM44L (exact)
}
_CW = {  # winding thermal mass
    'SPN_Z': 74.8 / 2, 'SPN_X': 74.8 / 2,      # SM85  (halved) = 37.40
    'HIP_Y': 74.8 / 2,                         # SM85  (halved) = 37.40
    'HIP_X': 39.95, 'THIGH_U': 39.95, 'SHIN_KNEE': 39.95,   # SM72  (exact)
    'SHIN_ANKLE': 74.77,                       # SM85  (exact)
    'TALUS': 19.03,                            # SM44L (exact)
}

# ---- structure thermal masses [J/K] (Node 2) ----
_CS = {
    'spine':   1040.7 / 2,   # 520.35  (halved)
    'pelvis':  1137.2 / 2,   # 568.60  (halved, rest of pelvis right side)
    'hip':     695.7,        # all structural + drivetrain + electronics
    'thigh_u': 859.6,        # structural + drivetrain + electronics
    'thigh_l': 709.3,        # RGA5-1 structural shells + PC+ABS caps
    'shin':    1743.7,       # structural + drivetrain + electronics + heatsink + fan
    'ankle':   322.7,        # talus: rest — drivetrain + electronics + housing
    'foot':    400.0,        # TODO no foot table yet
}

# ---- fabric / covering thermal masses [J/K] (Node 3) and structure->fabric R [K/W] ----
_CF = {'hip': 63.0, 'thigh_u': 535.3, 'thigh_l': 126.9, 'shin': 2118.4}
_RSTACK = {
    'hip':     2.35,         # measured [K/W]
    'thigh_u': 0.251,        # measured [K/W]
    'thigh_l': 1.520,        # measured
    'shin':    0.285,        # measured
}

CONFIG = LimbConfig(
    name='leg',
    ACTS=_ACTS,
    STRUCTS=_STRUCTS,
    FABRICS=['hip', 'thigh_u', 'thigh_l', 'shin'],   # segments with a Node 3 covering
    TOPO={                                            # (housing, output); housing = segment the motor sits in
        'SPN_Z':      ('spine', 'spine'),             # spine Z motor: leaf on spine (no link)
        'SPN_X':      ('spine', 'pelvis'),            # spine X motor: links spine -> pelvis
        'HIP_Y':      ('pelvis', 'hip'),              # links pelvis -> hip
        'HIP_X':      ('hip', 'thigh_u'),             # links hip -> thigh_u
        'THIGH_U':    ('thigh_u', 'thigh_l'),         # links thigh_u -> thigh_l
        'SHIN_KNEE':  ('shin', 'thigh_l'),            # knee motor in shin; links shin -> thigh_l
        'SHIN_ANKLE': ('shin', 'ankle'),              # ankle motor in shin; links shin -> ankle
        'TALUS':      ('ankle', 'foot'),              # links ankle(talus) -> foot
    },
    TS_COLS={                                         # TODO match the leg thermocouple CSV columns
        'spine': 'Ts_spine', 'pelvis': 'Ts_pelvis', 'hip': 'Ts_hip',
        'thigh_u': 'Ts_thigh_u', 'thigh_l': 'Ts_thigh_l', 'shin': 'Ts_shin',
        'ankle': 'Ts_ankle', 'foot': 'Ts_foot',
    },

    C_M=np.array([_CM[a] for a in _ACTS]),
    C_WIND=np.array([_CW[a] for a in _ACTS]),
    C_S_FIX=_CS,
    C_F_FIX=_CF,
    R_STACK=_RSTACK,
    AREA=_AREA,

    QFET=4.0,                                         # TODO confirm FET load
    EXTRA_Q={},                                       # TODO any constant electronics load

    BOUNDARY_ACTS=[],                                 # TODO no measured torso boundary yet; spine motors driven
    ENCLOSED={},                                      # every segment has a measured ambient area

    MOTOR={'SPN_Z': 'SM85', 'SPN_X': 'SM85', 'HIP_Y': 'SM85', 'HIP_X': 'SM72',
           'THIGH_U': 'SM72', 'SHIN_KNEE': 'SM72', 'SHIN_ANKLE': 'SM85', 'TALUS': 'SM44L'},

    # leg uses the MEASURED winding mass directly (scale 1.0); the arm used 2.5 as a fit fudge.
    DPDT_THRESH=5.0, CADAPT_TAU=80.0, ADAPT_C=True, C_WIND_SCALE=1.0,

    TRAIN=[],                                         # TODO data tags once collected
    TEST=[],
)
