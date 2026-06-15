"""Assemble time-aligned ARM datasets: motor T & P (lptn logs) + structure TCs (+1h).
Writes rom_1d/data/arm/data_<tag>.csv. Arm-specific raw-log parsing; a leg version
would be an analogous assemble_leg.py reading the leg test logs.

The GL860 thermocouple logger clock runs 1 h behind the actuator host, so TC
timestamps get +1 h to align with the motor logs.
"""
import pandas as pd, numpy as np, re, os
from scipy.signal import savgol_filter
from datetime import datetime, timedelta

DATA = '/Users/jiongchen/Documents/F03_actuator_temp/0604_3cases_testing_results'   # raw logs
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'arm')
os.makedirs(OUT, exist_ok=True)

ARM = ['left_shoulder_j1', 'left_shoulder_j2', 'left_upper_arm_twist', 'left_elbow',
       'left_wrist_roll', 'left_wrist_pitch', 'left_wrist_yaw']
SHORT = {'left_shoulder_j1': 'J1', 'left_shoulder_j2': 'J2', 'left_upper_arm_twist': 'TWIST',
         'left_elbow': 'ELBOW', 'left_wrist_roll': 'ROLL', 'left_wrist_pitch': 'PITCH',
         'left_wrist_yaw': 'YAW'}
CH = {'CH1': 'Ts_humerus_u', 'CH2': 'Ts_humerus_l', 'CH3': 'Ts_forearm_u', 'CH4': 'Ts_forearm_l',
      'CH5': 'Ts_wrist', 'CH6': 'Ts_hand', 'CH15': 'T_amb'}

RUNS = [  # (tag, actuator csv, thermocouple csv)
 ('i50',   'lptn_cal_20260604-144604_i50_wristNeckOverTemp.csv', '060426-134519.CSV'),
 ('i25a',  'lptn_cal_20260604-145657_i25_wristNeckOverTemp.csv', '060426-135627.CSV'),
 ('i25b',  'lptn_cal_20260604-153512_i25_wristNeckOverTemp_02.csv', '060426-143458.CSV'),
 ('hi19a', 'lptn_cal_20260605-141425_19_9A.csv', '060526-131403.CSV'),
 ('hi19b', 'lptn_cal_20260605-152432_19_9A_02.csv', '060526-142416.CSV'),
 ('depal_fast', 'lptn_cal_20260604-163624_jointCustomIq_02_fastSample.csv', '060426-153616.CSV'),
 ('depal_ss', 'lptn_cal_20260604-165542_jointCustomIq_02_continued_slowSample_untilSteadyState.csv', '060426-153616.CSV'),
 ('depal_sat', 'lptn_cal_20260604-173328_jointCustomIq_Saturated_torqueLogged.csv', '060426-153616.CSV'),
]


def load_tc(fn):
    txt = open(os.path.join(DATA, fn), encoding='utf-8', errors='replace').read().splitlines()
    di = [i for i, l in enumerate(txt) if l.strip() == 'Data'][0]
    S = pd.read_csv(os.path.join(DATA, fn), skiprows=di + 1); S = S[S['Date&Time'] != 'Time']
    S.index = pd.to_datetime(S['Date&Time'], format='%Y/%m/%d %H:%M:%S') + timedelta(hours=1)
    for c in CH:
        S[c] = pd.to_numeric(S[c], errors='coerce')
    return S[list(CH)].rename(columns=CH)


for tag, af, tf in RUNS:
    a = pd.read_csv(os.path.join(DATA, af), usecols=['t_rel', 'joint', 'Tmotor', 'P_motor'])
    s0 = datetime.strptime(re.search(r'(\d{8}-\d{6})', af).group(1), '%Y%m%d-%H%M%S')
    a = a[a['joint'].isin(ARM)].copy()
    a['clk'] = s0 + pd.to_timedelta(a['t_rel'], unit='s'); a = a.set_index('clk')
    if len(a) == 0:
        print(tag, "NO ARM JOINTS"); continue
    grid = pd.date_range(a.index.min().ceil('s'), a.index.max().floor('s'), freq='2s')

    def wide(col, pref):
        p = a.pivot_table(index=a.index, columns='joint', values=col)
        p = p.reindex(p.index.union(grid)).interpolate('time').reindex(grid)
        return p.rename(columns=lambda j: pref + SHORT[j])

    Tm = wide('Tmotor', 'Tm_'); P = wide('P_motor', 'P_')
    for j in ARM:
        if 'P_' + SHORT[j] not in P:
            P['P_' + SHORT[j]] = 0.0
        if 'Tm_' + SHORT[j] not in Tm:
            Tm['Tm_' + SHORT[j]] = np.nan
    S = load_tc(tf)
    Sg = S.reindex(S.index.union(grid)).interpolate('time').reindex(grid)
    df = pd.concat([Tm.sort_index(axis=1), P.sort_index(axis=1), Sg], axis=1)
    df.insert(0, 't_s', (df.index - df.index[0]).total_seconds())
    w = min(25, (len(df) // 2) * 2 - 1)
    if w >= 5:
        for c in df.columns:
            if c.startswith(('Tm_', 'Ts_')) or c == 'T_amb':
                if df[c].notna().all():
                    df[c] = savgol_filter(df[c].values, w, 2)
    df.to_csv(os.path.join(OUT, f'data_{tag}.csv'), index=False)
    nT = df.filter(like='Tm_').notna().any().sum()
    print(f"{tag:10s} rows={len(df):4d} dur={df['t_s'].iloc[-1]/60:5.1f}min  "
          f"armTm={nT}  TCnan={df.filter(like='Ts_').isna().sum().sum()}")
