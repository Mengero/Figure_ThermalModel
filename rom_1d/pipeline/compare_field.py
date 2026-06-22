"""Compare open-loop prediction accuracy across ALL field cases (UPS + depal).

For each case: take the longest continuous burst, resample, run the model fully open-loop
(battery->torso), mask non-physical readings, and tabulate per-joint RMSE. Outputs a summary
table + a grouped per-joint RMSE bar chart (figures/field_compare.png).
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import _common as C
sys.path.insert(0, C.ROOT)
from core.observer import ThermalObserver          # noqa: E402

SH = {'left_shoulder_j1': 'J1', 'left_shoulder_j2': 'J2', 'left_upper_arm_twist': 'TWIST',
      'left_elbow': 'ELBOW', 'left_wrist_roll': 'ROLL', 'left_wrist_pitch': 'PITCH', 'left_wrist_yaw': 'YAW'}
CASES = [   # label, path (rel to rom_1d), dt, amb_offset
    ('c_1081 UPS', 'data/ups/c_1081_ups_cleaned.csv', 4.0, 0.0),
    ('c_1261 UPS', 'data/ups/c_1261_ups_cleaned.csv', 10.0, 0.0),
    ('0517 UPS(idle)', 'data/ups/ups_0517_idle.csv', 10.0, 0.0),
    ('c_1092 depal', 'data/depal/c_1092_depal_cleaned.csv', 4.0, 0.0),
    ('c_1076 depal', 'data/depal/c_1076_depal.csv', 10.0, 0.0),
]


def load_longest(M, path, dt, amb_offset):
    p = path if os.path.exists(path) else os.path.join(C.ROOT, path)
    raw = pd.read_csv(p); raw['t'] = pd.to_datetime(raw['timestamp_utc'])
    raw = raw.sort_values('t')
    raw['J'] = raw['J'] if 'J' in raw.columns else raw['actuator_name'].map(SH)
    ts = raw['t'].drop_duplicates().sort_values()
    burst = (ts.diff().dt.total_seconds() > 120).cumsum()
    raw['burst'] = raw['t'].map(dict(zip(ts, burst)))
    dur = raw.groupby('burst')['t'].agg(lambda s: (s.max() - s.min()).total_seconds())
    d = raw[raw['burst'] == dur.idxmax()].copy()           # longest continuous segment
    t0 = d['t'].min(); d['s'] = (d['t'] - t0).dt.total_seconds()
    grid = np.arange(0, d['s'].max() + 1e-6, dt)
    df = pd.DataFrame({'t_s': grid}); R20 = M.cfg.R20
    for J in M.ACTS:
        g = d[d['J'] == J].sort_values('s')
        if g.empty:
            df['Tm_' + J] = np.nan; df['P_' + J] = 0.0; continue
        Tm = np.interp(grid, g['s'], g['Tmotor_degC'])
        P = (np.interp(grid, g['s'], g['real_power_W']) if 'real_power_W' in g.columns
             else 2.0 * np.interp(grid, g['s'], g['power_W']) * R20[J] * (234.5 + Tm) / 254.5)
        df['Tm_' + J] = Tm; df['P_' + J] = P
    ac = 'Tamb_filled' if 'Tamb_filled' in d.columns else 'Tambient_degC'
    aa = d.dropna(subset=[ac]).sort_values('s'); df['T_amb'] = (np.interp(grid, aa['s'], aa[ac]) + amb_offset) if len(aa) else 25.0
    bc = 'Tbatt_max_filled' if 'Tbatt_max_filled' in d.columns else ('Tbatt_max_degC' if 'Tbatt_max_degC' in d.columns else None)
    if bc:
        bb = d.dropna(subset=[bc]).sort_values('s')
        if len(bb): df['T_torso'] = np.interp(grid, bb['s'], bb[bc])
    return df, grid[-1] / 60.0


def main():
    M = C.rom('arm'); M.build(C.load_params('arm'))
    res = {}
    print('%-16s %6s  %s' % ('case', 'dur', '  '.join('%5s' % a for a in M.ACTS) + '   ALL'))
    for label, path, dt, off in CASES:
        df, dur = load_longest(M, path, dt, off)
        r = ThermalObserver(M, forced_drop=tuple(range(M.NM))).run(df, dt)
        pred = r['eff']; meas = r['meas'].copy()
        meas[(meas < 0) | (meas > 150)] = np.nan            # mask dead-sensor garbage (e.g. c_1076 YAW)
        rm = []
        for i in range(M.NM):
            e = pred[:, i] - meas[:, i]
            rm.append(np.sqrt(np.nanmean(e ** 2)) if np.isfinite(meas[:, i]).any() else np.nan)
        allr = np.sqrt(np.nanmean((pred - meas) ** 2))
        res[label] = rm
        print('%-16s %5.0fm  %s  %5.1f' % (label, dur, '  '.join(('%5.1f' % v if np.isfinite(v) else '  -- ') for v in rm), allr))
    fig, ax = plt.subplots(figsize=(13, 6)); x = np.arange(M.NM); w = 0.16
    for n, (label, rm) in enumerate(res.items()):
        ax.bar(x + n * w, [v if np.isfinite(v) else 0 for v in rm], w, label=label)
    ax.axhline(15, color='k', ls='--', lw=1, label='±15 C')
    ax.set_xticks(x + 2 * w); ax.set_xticklabels(M.ACTS); ax.set_ylabel('open-loop prediction RMSE [°C]')
    ax.set_title('Field prediction accuracy per joint — UPS vs depal cases'); ax.grid(axis='y', alpha=.3); ax.legend(fontsize=8)
    out = C.fig_path('field_compare.png'); plt.tight_layout(); plt.savefig(out, dpi=600); print('\nsaved', out)


if __name__ == '__main__':
    main()
