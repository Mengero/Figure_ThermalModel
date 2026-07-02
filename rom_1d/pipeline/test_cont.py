"""Held-out test of one fully-instrumented case (motors + structures) vs the model.

    python test_cont.py --limb arm --case 0616cont

Built for the continuous left-arm run (0616cont): both motor temps and structure
thermocouples are present, so it compares both. Uses the currently-saved fit
(fits/<limb>_fitted_params.json) — does NOT fit. Output: figures/<limb>_<case>_test.png
"""
import argparse
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import _common as C

ap = argparse.ArgumentParser()
ap.add_argument('--limb', default='arm')
ap.add_argument('--case', default='0616cont')
ap.add_argument('--dt', type=float, default=2.0)
a = ap.parse_args()

M = C.rom(a.limb); M.build(C.load_params(a.limb))
d = C.load(a.limb, a.case); X = M.simulate(d, a.dt); obs = M.struct_obs(X, d)
t = d['t_s'] / 60
order = M.ACTS; col = dict(zip(order, plt.cm.tab10.colors))
SCOL = {s: plt.cm.tab10.colors[i] for i, s in enumerate(M.TS_COLS)}

fig, ax = plt.subplots(2, 1, figsize=(13, 11), sharex=True)
me = []
for i, act in enumerate(order):
    m = d['Tm_' + act].to_numpy()
    if not np.isfinite(m).any():
        continue
    ax[0].plot(t, m, lw=1.6, color=col[act], label=act)
    ax[0].plot(t, X[:, i], '--', color=col[act], lw=1.2)
    me.append(X[:, i] - m)
se = []
for s, c in M.TS_COLS.items():
    if c not in d:
        continue
    ax[1].plot(t, d[c], lw=1.6, color=SCOL[s], label=s)
    ax[1].plot(t, obs[s], '--', color=SCOL[s], lw=1.2)
    se.append(obs[s] - d[c].to_numpy())
mR = np.sqrt(np.mean(np.concatenate(me) ** 2))
sR = np.sqrt(np.mean(np.concatenate(se) ** 2)) if se else float('nan')
ax[0].set_ylabel('motor T [°C]  (solid=meas, dashed=model)')
ax[0].set_title('%s — %s test (motor RMSE %.2f C, struct RMSE %.2f C)' % (a.limb, a.case, mR, sR))
ax[1].set_ylabel('structure T [°C]'); ax[1].set_xlabel('time [min]')
for x in ax:
    x.grid(alpha=.3); x.legend(fontsize=8, ncol=7, loc='lower right')
out = C.fig_path('%s_test.png' % a.case, a.limb); plt.tight_layout(); plt.savefig(out, dpi=600)
print('saved %s' % out)
print('motor RMSE = %.2f C   structure RMSE = %.2f C' % (mR, sR))
for i, act in enumerate(order):
    m = d['Tm_' + act].to_numpy()
    if np.isfinite(m).any():
        print('  motor  %-7s model %5.1f  meas %5.1f' % (act, X[-1, i], m[-1]))
for s, c in M.TS_COLS.items():
    if c in d:
        print('  struct %-7s model %5.1f  meas %5.1f' % (s, obs[s][-1], d[c].iloc[-1]))
