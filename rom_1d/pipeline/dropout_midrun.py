"""Mid-run single-sensor dropout: sweep each joint, drop its thermistor at a chosen
instant (default: right after a heavy depal lift), and show how the model's virtual
prediction tracks the withheld true reading afterward + the max error during later operation.

    python dropout_midrun.py                       # c_1092 depal, dropout @ 82 min
    python dropout_midrun.py --t0 82 --case data/depal/c_1092_depal_cleaned.csv --dt 4
"""
import argparse, sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import _common as C
sys.path.insert(0, C.ROOT)
from core.observer import ThermalObserver          # noqa: E402
from _faults import inject                          # noqa: E402
import compare_field as CF                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--case', default='data/depal/c_1092_depal_cleaned.csv')
ap.add_argument('--dt', type=float, default=4.0)
ap.add_argument('--t0', type=float, default=82.0)   # dropout instant [min], just after a heavy J2 lift
a = ap.parse_args()

M = C.rom('arm'); M.build(C.load_params('arm'))
df, dur = CF.load_longest(M, a.case, a.dt, 0.0)
t = df['t_s'].to_numpy() / 60.0
ACTS = M.ACTS
col = dict(zip(ACTS, plt.cm.tab10.colors))

rows = []
fig, axes = plt.subplots(4, 2, figsize=(15, 16), sharex=True)
axes = axes.ravel()
for j, J in enumerate(ACTS):
    i = ACTS.index(J)
    dff, true, _ = inject(df.copy(), J, 'nan', a.t0, dur + 1)   # drop from t0 to end
    r = ThermalObserver(M).run(dff, a.dt)
    virt = r['eff'][:, i]                                       # virtual reading while faulted
    post = t >= a.t0
    err = virt[post] - true[post]
    finite = np.isfinite(true[post])
    maxe = np.nanmax(np.abs(err[finite])) if finite.any() else np.nan
    ende = err[finite][-1] if finite.any() else np.nan
    rmse = np.sqrt(np.nanmean(err[finite] ** 2)) if finite.any() else np.nan
    rows.append((J, maxe, rmse, ende))

    ax = axes[j]
    ax.plot(t, true, lw=1.5, color=col[J], label='true (withheld)')
    ax.plot(t[post], virt[post], '--', lw=1.6, color='k', label='virtual (model)')
    ax.plot(t[~post], true[~post], lw=1.5, color=col[J], alpha=.35)   # pre-dropout (sensor live)
    ax.axvline(a.t0, color='r', ls=':', lw=1.2)
    ax.axvspan(a.t0, t[-1], color='r', alpha=.04)
    ax.set_title('%s dropout @ %.0f min   max err %.1f C   end err %+.1f C   RMSE %.1f C'
                 % (J, a.t0, maxe, ende, rmse), fontsize=10)
    ax.set_ylabel('motor T [C]'); ax.grid(alpha=.3); ax.legend(fontsize=8, loc='best')
axes[-1].axis('off') if len(ACTS) < len(axes) else None
for ax in axes[-2:]:
    ax.set_xlabel('time [min]')
plt.suptitle('Mid-run single-sensor dropout after a heavy load — virtual prediction vs withheld truth (%s)'
             % a.case.split('/')[-1], fontsize=13)
tag = a.case.split('/')[-1].replace('.csv', '')
out = C.fig_path('dropout_midrun_%s.png' % tag); plt.tight_layout(); plt.savefig(out, dpi=600)

print('dropout @ %.0f min  (run %.0f min, %.0f min of later operation)' % (a.t0, dur, dur - a.t0))
print('%-6s %10s %10s %10s' % ('joint', 'max|err|', 'RMSE', 'end err'))
for J, mx, rm, en in rows:
    print('%-6s %9.1f  %9.1f  %+9.1f' % (J, mx, rm, en))
print('\nsaved', out)
