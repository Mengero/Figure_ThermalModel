"""Open-loop prediction accuracy (experiment vs simulation) for a field case (UPS / depal).

The model is run fully open-loop (every motor predicted from power + ambient + battery-torso,
no thermistor feedback) and compared to the measured motor temps. This is the model's
field prediction accuracy — and the ceiling for the virtual-sensor / dropout study.

    python predict_field.py --csv data/ups/c_1081_ups_cleaned.csv --dt 4 --amb-offset 0
"""
import argparse, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import _common as C
sys.path.insert(0, C.ROOT)
from core.observer import ThermalObserver          # noqa: E402
import compare_field as CF                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--limb', default='arm')
ap.add_argument('--csv', required=True)
ap.add_argument('--dt', type=float, default=4.0)
ap.add_argument('--amb-offset', type=float, default=0.0)
a = ap.parse_args()

M = C.rom(a.limb); M.build(C.load_params(a.limb))
df, _ = CF.load_longest(M, a.csv, a.dt, a.amb_offset)   # longest continuous burst, field long-format
tag = os.path.splitext(os.path.basename(a.csv))[0]

# open-loop: force every motor "dropped" so eff = pure model prediction
r = ThermalObserver(M, forced_drop=tuple(range(M.NM))).run(df, a.dt)
t = r['t']; pred = r['eff']; meas = r['meas'].copy()
meas[(meas < 0) | (meas > 150)] = np.nan                # mask non-physical readings (dead sensors)
order = M.ACTS; col = dict(zip(order, plt.cm.tab10.colors))

fig, ax = plt.subplots(3, 1, figsize=(13, 14), sharex=True)
rmse = {}
for i, J in enumerate(order):
    m = meas[:, i]
    if not np.isfinite(m).any():
        continue
    e = pred[:, i] - m
    rmse[J] = np.sqrt(np.nanmean(e ** 2))
    ax[0].plot(t, m, lw=1.4, color=col[J], label='%s (RMSE %.1f)' % (J, rmse[J]))
    ax[0].plot(t, pred[:, i], '--', lw=1.1, color=col[J])
    ax[1].plot(t, e, lw=1.0, color=col[J], label=J)
    ax[2].plot(t, df['P_' + J], lw=1.0, color=col[J], label=J)
if 'T_torso' in df:
    ax[0].plot(t, df['T_torso'], 'k:', lw=1.2, label='battery/torso')
ax[0].plot(t, df['T_amb'], color='gray', ls=':', lw=1.2, label='ambient')
ax[1].axhline(0, color='k', lw=.5)
allr = np.sqrt(np.nanmean((pred - meas)[:, [i for i in range(M.NM) if np.isfinite(meas[:, i]).any()]] ** 2))
ax[0].set_ylabel('motor T [°C] (solid=exp, dashed=sim)')
ax[0].set_title('%s — open-loop prediction vs experiment (overall motor RMSE %.2f °C)' % (tag, allr))
ax[0].grid(alpha=.3); ax[0].legend(fontsize=8, ncol=4, loc='upper left')
ax[1].set_ylabel('sim − exp [°C]'); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8, ncol=7)
ax[2].set_ylabel('motor copper power [W]'); ax[2].set_xlabel('time [min]'); ax[2].grid(alpha=.3); ax[2].legend(fontsize=8, ncol=7)
out = C.fig_path('%s_predict.png' % tag); plt.tight_layout(); plt.savefig(out, dpi=600)
print('overall motor RMSE = %.2f °C  ->  saved %s' % (allr, out))
for J in order:
    if J in rmse:
        e = pred[:, order.index(J)] - meas[:, order.index(J)]
        print('  %-6s RMSE %5.2f  max %5.1f  (sim end %5.1f, exp end %5.1f)'
              % (J, rmse[J], np.nanmax(np.abs(e)), pred[-1, order.index(J)], meas[-1, order.index(J)]))
