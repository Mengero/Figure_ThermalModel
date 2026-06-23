"""Zero-survivor case: ALL thermistors dropped from boot. The model runs fully open-loop,
anchored only by ambient + the battery/torso boundary, and every node is initialized from the
battery temperature (the only known body temperature). Scores each motor's prediction vs the
withheld true reading over the whole run.

    python dropout_allzero.py                                # c_1092 depal
    python dropout_allzero.py --case data/ups/c_1081_ups_cleaned.csv --dt 4
"""
import argparse, sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import _common as C
sys.path.insert(0, C.ROOT)
from core.observer import ThermalObserver          # noqa: E402
import compare_field as CF                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--case', default='data/depal/c_1092_depal_cleaned.csv')
ap.add_argument('--dt', type=float, default=4.0)
a = ap.parse_args()

M = C.rom('arm'); M.build(C.load_params('arm'))
df, dur = CF.load_longest(M, a.case, a.dt, 0.0)
t = df['t_s'].to_numpy() / 60.0
ACTS = M.ACTS
col = dict(zip(ACTS, plt.cm.tab10.colors))
true_all = df[['Tm_' + a_ for a_ in ACTS]].to_numpy(dtype=float)
true_all[(true_all < 0) | (true_all > 150)] = np.nan        # mask dead-sensor garbage (e.g. c_1076 YAW)
Tbatt = df['T_torso'].to_numpy() if 'T_torso' in df else np.full(len(df), M.torso_temp)

r = ThermalObserver(M, forced_drop=tuple(range(M.NM))).run(df, a.dt)   # all dropped
print('ALL 7 dropped, seeded from battery T0=%.1f C  (%s, run %.0f min)' % (Tbatt[0], a.case.split('/')[-1], dur))
print('%-6s %10s %10s %10s' % ('joint', 'max|err|', 'RMSE', 'end err'))

fig, axes = plt.subplots(4, 2, figsize=(15, 16), sharex=True); axes = axes.ravel()
rows = []
for j, J in enumerate(ACTS):
    true = true_all[:, j]; virt = r['eff'][:, j]; ok = np.isfinite(true)
    if not ok.any():                                        # dead sensor: no truth to score against
        rows.append((J, np.nan, np.nan, np.nan)); print('%-6s %9s  %9s  %9s  (dead sensor)' % (J, '--', '--', '--'))
        ax = axes[j]; ax.plot(t, virt, '--', lw=1.6, color='k', label='virtual (open-loop)')
        ax.plot(t, Tbatt, ':', lw=1.1, color='tab:brown', label='battery/torso')
        ax.set_title('%s   (dead thermistor — prediction only)' % J, fontsize=10)
        ax.set_ylabel('motor T [C]'); ax.grid(alpha=.3); ax.legend(fontsize=8, loc='best'); continue
    e = virt - true; maxe = np.nanmax(np.abs(e[ok])); rmse = np.sqrt(np.nanmean(e[ok] ** 2))
    rows.append((J, maxe, rmse, e[ok][-1]))
    print('%-6s %9.1f  %9.1f  %+9.1f' % (J, maxe, rmse, e[ok][-1]))
    ax = axes[j]
    ax.plot(t, true, lw=1.5, color=col[J], label='true (withheld)')
    ax.plot(t, virt, '--', lw=1.6, color='k', label='virtual (open-loop)')
    ax.plot(t, Tbatt, ':', lw=1.1, color='tab:brown', label='battery/torso')
    ax.set_title('%s   max err %.1f C   RMSE %.1f C' % (J, maxe, rmse), fontsize=10)
    ax.set_ylabel('motor T [C]'); ax.grid(alpha=.3); ax.legend(fontsize=8, loc='best')
if len(ACTS) < len(axes):
    axes[-1].axis('off')
for ax in axes[-2:]:
    ax.set_xlabel('time [min]')
mean_rmse = np.nanmean([r2 for _, _, r2, _ in rows]); worst_max = np.nanmax([mx for _, mx, _, _ in rows])
plt.suptitle('ALL thermistors dropped — open-loop prediction (init from battery %.1f C) — %s   mean RMSE %.1f C'
             % (Tbatt[0], a.case.split('/')[-1], mean_rmse), fontsize=13)
tag = a.case.split('/')[-1].replace('.csv', '')
out = C.fig_path('dropout_allzero_%s.png' % tag); plt.tight_layout(); plt.savefig(out, dpi=600)
print('\nmean RMSE %.1f C   worst max %.1f C\nsaved %s' % (mean_rmse, worst_max, out))
