"""Lower-limb LimbConfig — masses/areas loaded from measured part tables.

Chain (proximal -> distal):  spine -> pelvis -> hip -> thigh_u -> thigh_l -> shin
-> ankle(talus) -> foot.  Runs on the SAME engine (core.ThermalROM) as the arm.

Motor inventory (8 motors), by the segment each is housed in:
  spine   : SPN_Z, SPN_X        (2x SM85)        — halved (centerline, shared L/R)
  pelvis  : HIP_Y               (SM85)           — exact (per-leg)
  hip     : HIP_X               (SM72)           — exact
  thigh_u : THIGH_U             (SM72)           — exact
  thigh_l : (none — passive segment)
  shin    : SHIN_KNEE (SM85), SHIN_ANKLE (SM72)  — exact
  ankle   : TALUS               (SM44L)          — exact
Covered (fabric) segments: hip, thigh_u, thigh_l, shin (each has a Node 3 covering).
Bare segments: spine, pelvis, ankle, foot.

AREA is measured per segment; spine & pelvis areas are halved in code (/2) per the
left/right symmetry instruction (centerline, shared L/R); hip and below are exact.

================================ TODО / FILL IN ================================
  * C_S_FIX['foot'] — set to 1324 J/K from FOOT ASSEMBLY table; foot houses no motor.
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
    'spine':   0.0190 / 2,   #  38,040 / 4 mm^2
    'pelvis':  0.0575 / 2,   #  57,502 / 2 mm^2
    'hip':     0.0273 / 2,   #  27,292 / 2 mm^2
    'thigh_u': 0.1199,       #  upper thigh
    'thigh_l': 0.0281,       #  lower thigh / knee
    'shin':    0.0661,       #  lower leg / shin
    'ankle':   0.0093,
    'foot':    0.0440,
}

# 8 real motors, ordered by housing segment (proximal -> distal)
_ACTS = ['SPN_Z', 'SPN_X', 'HIP_Y', 'HIP_X', 'THIGH_U', 'SHIN_KNEE', 'SHIN_ANKLE', 'TALUS']

# ---- motor thermal masses [J/K]  (assembly = Node 1 ; winding = Winding TM) ----
# only the spine SM85s (SPN_Z, SPN_X) are HALVED (centerline, shared L/R); HIP_Y is
# per-leg so it's EXACT, and hip and below are EXACT.
_CM = {  # assembly thermal mass [J/K] — RE-MEASURED motor mass (SM85=491.2, SM72=158.35, SM44L=144.22)
    'SPN_Z': 491.2 / 2, 'SPN_X': 491.2 / 2,    # SM85 spine (halved, centerline) = 245.6
    'HIP_Y': 491.2,                            # SM85  (per-leg)
    'HIP_X': 158.35, 'THIGH_U': 158.35, 'SHIN_ANKLE': 158.35,  # SM72
    'SHIN_KNEE': 491.2,                        # SM85 knee
    'TALUS': 144.22,                           # SM44L ankle_x
}
_CW = {  # winding thermal mass [J/K] — UNCHANGED (same as before)
    'SPN_Z': 74.8 / 2, 'SPN_X': 74.8 / 2,      # SM85  (halved) = 37.40
    'HIP_Y': 74.8,                             # SM85  (per-leg)
    'HIP_X': 39.95, 'THIGH_U': 39.95, 'SHIN_ANKLE': 39.95,  # SM72
    'SHIN_KNEE': 74.77,                        # SM85 knee
    'TALUS': 19.03,                            # SM44L ankle_x
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
    'foot':    1324.0,       # FOOT ASSEMBLY total (Al 405 + steel 110 + Cu 64 + polymers 400 + rubber 345)
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
    TOPO={                                            # (housing, output): motor<->housing via R2, housing<->output via R_link
        'SPN_Z':      ('spine', 'spine'),             # R2->spine; leaf (torso boundary via R_torso, see TORSO)
        'SPN_X':      ('pelvis', 'spine'),            # R2->pelvis; R_link pelvis<->spine
        'HIP_Y':      ('hip', 'pelvis'),              # R2->hip;    R_link hip<->pelvis
        'HIP_X':      ('thigh_u', 'hip'),             # R2->thigh_u; R_link thigh_u<->hip
        'THIGH_U':    ('thigh_u', 'thigh_l'),         # = hip_z; R2->thigh_u; R_link thigh_u<->thigh_l
        'SHIN_KNEE':  ('shin', 'thigh_l'),            # knee motor in shin; R2->shin; R_link shin<->thigh_l
        'SHIN_ANKLE': ('shin', 'ankle'),              # R2->shin;  R_link shin<->ankle
        'TALUS':      ('ankle', 'foot'),              # ankle motor; R2->ankle; R_link ankle<->foot
    },
    TS_COLS={                                         # GL860 TC -> segment (foot has no TC; see assemble_leg.py)
        'spine': 'Ts_spine', 'pelvis': 'Ts_pelvis', 'hip': 'Ts_hip',
        'thigh_u': 'Ts_thigh_u', 'thigh_l': 'Ts_thigh_l', 'shin': 'Ts_shin',
        'ankle': 'Ts_ankle',
    },

    C_M=np.array([_CM[a] for a in _ACTS]),
    C_WIND=np.array([_CW[a] for a in _ACTS]),
    C_S_FIX=_CS,
    C_F_FIX=_CF,
    R_STACK=_RSTACK,
    AREA=_AREA,

    QFET=4.0,                                         # TODO confirm FET load
    EXTRA_Q={},                                       # TODO any constant electronics load

    BOUNDARY_ACTS=[],                                 # SPN_Z predicted (not clamped); torso is its sink (see TORSO)
    ENCLOSED={},                                      # every segment has a measured ambient area
    TORSO=('SPN_Z', 40.0),                            # SPN_Z sinks to torso via fitted R_torso; battery cell temp (arm treatment): field cases pull T_torso from Tbatt_max via fields.py; 40 C fallback for benchtop

    MOTOR={'SPN_Z': 'SM85', 'SPN_X': 'SM85', 'HIP_Y': 'SM85', 'HIP_X': 'SM72',
           'THIGH_U': 'SM72', 'SHIN_KNEE': 'SM85', 'SHIN_ANKLE': 'SM72', 'TALUS': 'SM44L'},

    R20={'SPN_Z': 0.078, 'SPN_X': 0.078, 'HIP_Y': 0.078, 'HIP_X': 0.096,   # phase resistance @20C [ohm], by motor size
         'THIGH_U': 0.096, 'SHIN_KNEE': 0.078, 'SHIN_ANKLE': 0.096, 'TALUS': 0.72},   # SM85=0.078, SM72=0.096, SM44L=0.72

    # same motor-C strategy as the arm: adaptive dip on |dP/dt| with C_WIND_SCALE=2.5.
    # NOTE: 2.5 was hand-tuned on arm data; revisit once leg data exists (leg C_WIND is measured).
    DPDT_THRESH=5.0, CADAPT_TAU=80.0, ADAPT_C=True, C_WIND_SCALE=2.5,

    TIE_R2=True,                                      # leg R2 tied per motor size (one fitted value per size; not pinned to provided values)
    FIT_RSTACK=True,                                  # fit structure->fabric R_STACK (else fixed measured values)
    HALVED_ACTS=('SPN_Z', 'SPN_X'),                   # spine motors halved (centerline): C/2, power/2, area/2 -> R2 x2
    STEADY_CASE='steady_c1081',                       # fold c_1081 steady point into the fit (constrains R_torso), same as arm
    TRAIN=['hi19a', 'depal_ss', 'i25b'],              # high-power + near-steady + TALUS-heating spread
    TEST=['i50', 'i25a', 'hi19b', 'depal_fast', 'depal_sat'],
)
