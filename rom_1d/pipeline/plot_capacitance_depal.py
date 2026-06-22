"""Adaptive motor capacitance C(t) for the depal case, zoomed to the first 10 min.

Shows how each motor's thermal capacitance dips from full-assembly C_M toward winding C_WIND
on every lift (|dP/dt| spike) and tries to recover with tau=CADAPT_TAU. Top: C(t); bottom:
the copper power driving it.

    python plot_capacitance_depal.py
"""
import sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import _common as C
sys.path.insert(0, C.ROOT)
import compare_field as CF                          # noqa: E402

LABEL, PATH, DT, OFF = 'c_1092 depal', 'data/depal/c_1092_depal_cleaned.csv', 4.0, 0.0
TMAX = 10.0   # minutes

M = C.rom('arm'); M.build(C.load_params('arm'))
col = dict(zip(M.ACTS, plt.cm.tab10.colors))

df, dur = CF.load_longest(M, PATH, DT, OFF)
t = df['t_s'].to_numpy() / 60.0
Cser = M.cmotor_series(df)                          # (n, NM) per-step motor C [J/K]
P = np.nan_to_num(df[['P_' + a for a in M.ACTS]].to_numpy())

fig, (aC, aP) = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
for i, J in enumerate(M.ACTS):
    aC.plot(t, Cser[:, i], lw=1.4, color=col[J],
            label='%s (C_M %.0f, C_w %.0f)' % (J, M.C_M[i], M.C_WIND[i]))
    aP.plot(t, P[:, i], lw=1.1, color=col[J], label=J)
aC.set_title('%s — adaptive motor capacitance, first %.0f min' % (LABEL, TMAX))
aC.set_ylabel('motor C [J/K]'); aC.grid(alpha=.3); aC.legend(fontsize=8, ncol=2)
aP.set_ylabel('motor copper power [W]'); aP.set_xlabel('time [min]')
aP.grid(alpha=.3); aP.legend(fontsize=8, ncol=4)
aC.set_xlim(0, TMAX)
out = C.fig_path('capacitance_depal_first10min.png'); plt.tight_layout(); plt.savefig(out, dpi=600)
print('saved', out)
