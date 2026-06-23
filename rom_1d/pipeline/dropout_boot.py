"""Boot-time single-sensor dropout: sweep each joint, drop its thermistor from t=0 (never
available), and score the model's virtual prediction against the withheld true reading over
the WHOLE run. Unlike the mid-run case there is no good reading to seed from, so the boot
logic initializes the faulted motor from a neighbouring structure.

    python dropout_boot.py                                   # c_1092 depal
    python dropout_boot.py --case data/ups/c_1081_ups_cleaned.csv --dt 4
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

rows = []
fig, axes = plt.subplots(4, 2, figsize=(15, 16), sharex=True)
axes = axes.ravel()
for j, J in enumerate(ACTS):
    i = ACTS.index(J)
    r = ThermalObserver(M, forced_drop=[J]).run(df, a.dt)   # dropped from boot
    virt = r['eff'][:, i]; true = true_all[:, i]
    ok = np.isfinite(true)
    err = virt[ok] - true[ok]
    maxe = np.nanmax(np.abs(err)); rmse = np.sqrt(np.nanmean(err ** 2))
    boot_e = err[0]; end_e = err[-1]
    rows.append((J, maxe, rmse, boot_e, end_e))

    ax = axes[j]
    ax.plot(t, true, lw=1.5, color=col[J], label='true (withheld)')
    ax.plot(t, virt, '--', lw=1.6, color='k', label='virtual (model, dropped @ boot)')
    ax.set_title('%s dropped @ boot   max err %.1f C   RMSE %.1f C   boot err %+.1f   end err %+.1f'
                 % (J, maxe, rmse, boot_e, end_e), fontsize=9.5)
    ax.set_ylabel('motor T [C]'); ax.grid(alpha=.3); ax.legend(fontsize=8, loc='best')
if len(ACTS) < len(axes):
    axes[-1].axis('off')
for ax in axes[-2:]:
    ax.set_xlabel('time [min]')
plt.suptitle('Boot-time single-sensor dropout — virtual prediction vs withheld truth (%s)'
             % a.case.split('/')[-1], fontsize=13)
tag = a.case.split('/')[-1].replace('.csv', '')
out = C.fig_path('dropout_boot_%s.png' % tag); plt.tight_layout(); plt.savefig(out, dpi=600)

print('boot dropout over full %.0f min run' % dur)
print('%-6s %10s %10s %10s %10s' % ('joint', 'max|err|', 'RMSE', 'boot err', 'end err'))
for J, mx, rm, be, en in rows:
    print('%-6s %9.1f  %9.1f  %+9.1f  %+9.1f' % (J, mx, rm, be, en))
print('\nsaved', out)
