"""Field-data loading: turn a robot long-format CSV into the per-step DataFrame the
ROM/observer consume (columns: t_s, P_<act>, Tm_<act>, T_amb, optional T_torso).

`load_field` takes the LONGEST continuous burst (gaps > 2 min split bursts), resamples
onto a uniform dt grid, and computes per-actuator copper power. The CSV stores
`power_W = 3/4*iq^2` (no resistance); the real copper loss is

    P = 1.5*iq^2*R(T) = 2*power_W * R20 * (234.5 + Tmotor)/(234.5 + 20)

so resistance + the copper temperature coefficient are applied here from the measured
motor temperature. This is the single field loader used by every analysis tool.
"""
import os
import numpy as np
import pandas as pd
import _common as C

# robot actuator_name -> ROM joint key (CSVs may already carry a 'J' column)
SH = {'left_shoulder_j1': 'J1', 'left_shoulder_j2': 'J2', 'left_upper_arm_twist': 'TWIST',
      'left_elbow': 'ELBOW', 'left_wrist_roll': 'ROLL', 'left_wrist_pitch': 'PITCH',
      'left_wrist_yaw': 'YAW'}


def load_field(M, path, dt, amb_offset=0.0):
    """Return (df, duration_min) for the longest continuous burst in a field CSV."""
    p = path if os.path.exists(path) else os.path.join(C.ROOT, path)
    raw = pd.read_csv(p); raw['t'] = pd.to_datetime(raw['timestamp_utc'])
    raw = raw.sort_values('t')
    raw['J'] = raw['J'] if 'J' in raw.columns else raw['actuator_name'].map(SH)
    ts = raw['t'].drop_duplicates().sort_values()
    burst = (ts.diff().dt.total_seconds() > 120).cumsum()
    raw['burst'] = raw['t'].map(dict(zip(ts, burst)))
    dur = raw.groupby('burst')['t'].agg(lambda s: (s.max() - s.min()).total_seconds())
    d = raw[raw['burst'] == dur.idxmax()].copy()              # longest continuous segment
    t0 = d['t'].min(); d['s'] = (d['t'] - t0).dt.total_seconds()
    grid = np.arange(0, d['s'].max() + 1e-6, dt)
    df = pd.DataFrame({'t_s': grid}); R20 = M.cfg.R20
    for J in M.ACTS:
        g = d[d['J'] == J].sort_values('s')
        if g.empty:
            df['Tm_' + J] = np.nan; df['P_' + J] = 0.0; continue
        Tm = np.interp(grid, g['s'], g['Tmotor_degC'])
        P = 2.0 * np.interp(grid, g['s'], g['power_W']) * R20[J] * (234.5 + Tm) / 254.5
        df['Tm_' + J] = Tm; df['P_' + J] = P
    ac = 'Tamb_filled' if 'Tamb_filled' in d.columns else 'Tambient_degC'
    aa = d.dropna(subset=[ac]).sort_values('s')
    df['T_amb'] = (np.interp(grid, aa['s'], aa[ac]) + amb_offset) if len(aa) else 25.0
    bc = ('Tbatt_max_filled' if 'Tbatt_max_filled' in d.columns
          else ('Tbatt_max_degC' if 'Tbatt_max_degC' in d.columns else None))
    if bc:
        bb = d.dropna(subset=[bc]).sort_values('s')
        if len(bb):
            df['T_torso'] = np.interp(grid, bb['s'], bb[bc])
    return df, grid[-1] / 60.0
