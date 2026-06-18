"""Left-arm LimbConfig — 18 states (7 motor + 7 structure + 4 fabric).

Chain: J1(shoulder) -> J2 -> TWIST -> ELBOW -> ROLL -> PITCH -> YAW(hand).
J1 is a driven boundary (clamped to its measured temperature); the shoulder is
enclosed in the torso shell (no ambient path) and inherits humerus_u's metal temp.
Hand low-voltage electronics add 4.36 W on YAW.  All masses are measured
(Node1 = motor assembly, Node2 = structure, Node3 = fabric); only R2, R_link, b are fit.
"""
import numpy as np
from core import LimbConfig

CONFIG = LimbConfig(
    name='arm',
    ACTS=['J1', 'J2', 'TWIST', 'ELBOW', 'ROLL', 'PITCH', 'YAW'],
    STRUCTS=['shoulder', 'hum_u', 'hum_l', 'fore_u', 'fore_l', 'wrist', 'hand'],
    FABRICS=['hum_u', 'hum_l', 'fore_u', 'fore_l'],
    TOPO={'J1': ('shoulder', 'shoulder'), 'J2': ('shoulder', 'hum_u'),
          'TWIST': ('hum_l', 'hum_u'), 'ELBOW': ('fore_u', 'hum_l'),
          'ROLL': ('fore_l', 'fore_u'), 'PITCH': ('wrist', 'fore_l'),
          'YAW': ('hand', 'wrist')},
    TS_COLS={'hum_u': 'Ts_humerus_u', 'hum_l': 'Ts_humerus_l', 'fore_u': 'Ts_forearm_u',
             'fore_l': 'Ts_forearm_l', 'wrist': 'Ts_wrist', 'hand': 'Ts_hand'},

    # motor full-assembly C [J/K] (Node 1). J1 (Jade SM72) assembly not yet measured.
    C_M=np.array([0.2529 * 430.0, 203.9, 203.9, 203.9, 114.16, 138.55, 155.12]),
    # winding-only C [J/K]; scaled by C_WIND_SCALE in the engine. J1 is a boundary (no transition).
    C_WIND=np.array([0.2529 * 430.0, 22.11, 22.11, 22.11, 10.40, 10.35, 10.40]),
    C_S_FIX={'shoulder': 676.0, 'hum_u': 222.3, 'hum_l': 778.1, 'fore_u': 1218.1,
             'fore_l': 399.7, 'wrist': 269.6 * 0.2, 'hand': 819.7 * 0.3},   # Node 2
    C_F_FIX={'hum_u': 61.5, 'hum_l': 6.2, 'fore_u': 32.8, 'fore_l': 50.2},   # Node 3 (fabric)

    R_STACK={'hum_u': 1.749, 'hum_l': 0.227, 'fore_u': 1.319, 'fore_l': 1.913},
    AREA={'shoulder': 0.015875, 'hum_u': 0.047561 - 0.015875, 'hum_l': 0.039904,
          'fore_u': 0.026855, 'fore_l': 0.027882, 'wrist': 0.015730, 'hand': 0.049},

    QFET=4.0,
    EXTRA_Q={'YAW': 4.36},                 # hand low-voltage electronics

    BOUNDARY_ACTS=[],                      # J1 now predicted (no clamp); torso is its heat sink
    ENCLOSED={'shoulder': 'hum_u'},        # no TC; inherits humerus_u metal at init
    TORSO=('J1', 40.0),                    # J1 motor sinks to a 40 C torso via fitted R_torso (its mount)

    DPDT_THRESH=5.0, CADAPT_TAU=80.0, ADAPT_C=True, C_WIND_SCALE=2.5,

    # per-phase winding resistance at 20 C [ohm] (for continuous-current-limit back-calc)
    R20={'J1': 0.096, 'J2': 0.098, 'TWIST': 0.098, 'ELBOW': 0.098,
         'ROLL': 0.45, 'PITCH': 0.45, 'YAW': 0.45},
    MOTOR={'J1': 'Jade SM72', 'J2': 'Citrine SM67', 'TWIST': 'Citrine SM67',
           'ELBOW': 'Citrine SM67', 'ROLL': 'Papaya SM44S', 'PITCH': 'Papaya SM44S',
           'YAW': 'Papaya SM44S'},

    TRAIN=['i50', 'i25a', 'depal_ss'],
    # hi19a/b are 3-joint endurance runs validated separately (see pipeline/plot_hi19.py)
    # 0616cont = continuous left-arm (motors+structures), held out as a TEST case
    TEST=['depal_fast', 'i25b', 'depal_sat', 'hi19a', 'hi19b', '0616cont'],
)
