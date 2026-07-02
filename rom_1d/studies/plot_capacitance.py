"""Adaptive motor capacitance C(t) vs the copper power that drives it.

Shows each motor's capacitance dipping from full assembly C_M toward the winding floor
C_WIND on every |dP/dt| transient, recovering with tau=CADAPT_TAU.

  python studies/plot_capacitance.py                       # depal + ups side by side (full runs)
  python studies/plot_capacitance.py --cases data/depal/c_1092_depal_cleaned.csv --tmax 10
                                                           # one case, zoomed to first 10 min

Output: figures/capacitance_vs_time.png (or capacitance_<case>_first<TMAX>min.png when zoomed).
"""
import argparse, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.join(os.path.dirname(HERE), 'pipeline'); sys.path.insert(0, PIPE)
import _common as C                                   # noqa: E402
from fields import load_field                          # noqa: E402

DEFAULT_CASES = ['data/depal/c_1092_depal_cleaned.csv', 'data/ups/c_1081_ups_cleaned.csv']

ap = argparse.ArgumentParser()
ap.add_argument('--cases', nargs='+', default=DEFAULT_CASES)
ap.add_argument('--dt', type=float, default=4.0)
ap.add_argument('--tmax', type=float, default=None, help='zoom to first TMAX minutes')
a = ap.parse_args()

M = C.rom('arm'); M.build(C.load_params('arm'))
col = dict(zip(M.ACTS, plt.cm.tab10.colors))
ncol = len(a.cases)
fig, axes = plt.subplots(2, ncol, figsize=(7 * ncol, 10), squeeze=False, sharex='col')
for j, case in enumerate(a.cases):
    df, dur = load_field(M, case, a.dt, 0.0)
    t = df['t_s'].to_numpy() / 60.0
    Cser = M.cmotor_series(df)
    P = np.nan_to_num(df[['P_' + a_ for a_ in M.ACTS]].to_numpy())
    aC, aP = axes[0, j], axes[1, j]
    for i, J in enumerate(M.ACTS):
        aC.plot(t, Cser[:, i], lw=1.3, color=col[J],
                label='%s (C_M %.0f, C_w %.0f)' % (J, M.C_M[i], M.C_WIND[i]))
        aP.plot(t, P[:, i], lw=1.0, color=col[J], label=J)
    label = os.path.basename(case).replace('.csv', '')
    ttl = '%s — adaptive motor capacitance%s' % (label, '' if a.tmax is None else ', first %.0f min' % a.tmax)
    aC.set_title(ttl); aC.set_ylabel('motor C [J/K]'); aC.grid(alpha=.3); aC.legend(fontsize=7, ncol=2)
    aP.set_ylabel('motor copper power [W]'); aP.set_xlabel('time [min]'); aP.grid(alpha=.3); aP.legend(fontsize=7, ncol=4)
    if a.tmax is not None:
        aC.set_xlim(0, a.tmax)
if a.tmax is not None and ncol == 1:
    out = C.fig_path('capacitance_%s_first%.0fmin.png' % (os.path.basename(a.cases[0]).replace('.csv', ''), a.tmax), 'studies')
else:
    out = C.fig_path('capacitance_vs_time.png', 'studies')
plt.tight_layout(); plt.savefig(out, dpi=600); print('saved', out)
