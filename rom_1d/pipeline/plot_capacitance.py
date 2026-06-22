"""Plot the per-motor adaptive capacitance C(t) = cmotor_series(df) for field cases.

Shows how each motor's thermal capacitance dips from full-assembly C_M toward winding C_WIND
when |dP/dt| spikes (sharp lifts) and recovers with tau=CADAPT_TAU. One column per case
(a depal case and a UPS case), one curve per actuator.

    python plot_capacitance.py            # default: c_1092 depal + c_1081 UPS
"""
import sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import _common as C
sys.path.insert(0, C.ROOT)
import compare_field as CF                          # noqa: E402

CASES = [   # label, path, dt, amb_offset
    ('c_1092 depal', 'data/depal/c_1092_depal_cleaned.csv', 4.0, 0.0),
    ('c_1081 UPS',   'data/ups/c_1081_ups_cleaned.csv',     4.0, 0.0),
]

M = C.rom('arm'); M.build(C.load_params('arm'))
col = dict(zip(M.ACTS, plt.cm.tab10.colors))

fig, axes = plt.subplots(2, len(CASES), figsize=(7 * len(CASES), 10), sharex='col')
for j, (label, path, dt, off) in enumerate(CASES):
    df, dur = CF.load_longest(M, path, dt, off)
    t = df['t_s'].to_numpy() / 60.0
    Cser = M.cmotor_series(df)                      # (n, NM) per-step motor C [J/K]
    P = np.nan_to_num(df[['P_' + a for a in M.ACTS]].to_numpy())
    aC, aP = axes[0, j], axes[1, j]
    for i, J in enumerate(M.ACTS):
        aC.plot(t, Cser[:, i], lw=1.3, color=col[J],
                label='%s (C_M %.0f, C_w %.0f)' % (J, M.C_M[i], M.C_WIND[i]))
        aP.plot(t, P[:, i], lw=1.0, color=col[J], label=J)
    aC.set_title('%s — adaptive motor capacitance' % label)
    aC.set_ylabel('motor C [J/K]'); aC.grid(alpha=.3); aC.legend(fontsize=7, ncol=2)
    aP.set_ylabel('motor copper power [W]'); aP.set_xlabel('time [min]')
    aP.grid(alpha=.3); aP.legend(fontsize=7, ncol=4)
out = C.fig_path('capacitance_vs_time.png'); plt.tight_layout(); plt.savefig(out, dpi=600)
print('saved', out)
