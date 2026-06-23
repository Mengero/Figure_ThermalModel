"""Worst-case multi-dropout: ONLY ONE actuator keeps its thermistor, the other six drop out
from boot. Sweep which actuator survives (7 scenarios). For each, the six dropped motors are
pure model predictions (anchored by the one survivor + ambient + torso); score them vs the
withheld true readings over the whole run.

    python dropout_solo.py                                   # c_1092 depal, boot
    python dropout_solo.py --case data/ups/c_1081_ups_cleaned.csv --dt 4
    python dropout_solo.py --t0 82                           # mid-run variant (survivor clamped,
                                                             #   others drop at t0)
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
ap.add_argument('--t0', type=float, default=0.0)    # 0 = boot dropout; >0 = mid-run dropout instant
ap.add_argument('--survivor', default=None)          # if set, also plot the per-joint time series
a = ap.parse_args()

M = C.rom('arm'); M.build(C.load_params('arm'))
df, dur = CF.load_longest(M, a.case, a.dt, 0.0)
t = df['t_s'].to_numpy() / 60.0
ACTS = M.ACTS; NM = M.NM
true_all = df[['Tm_' + a_ for a_ in ACTS]].to_numpy(dtype=float)

# RMSE[survivor, dropped] and max-err
RMSE = np.full((NM, NM), np.nan); MAXE = np.full((NM, NM), np.nan)
rows = []
for si, S in enumerate(ACTS):
    dropped = [j for j in range(NM) if j != si]
    if a.t0 <= 0:                                   # boot: drop six from t=0
        r = ThermalObserver(M, forced_drop=dropped).run(df, a.dt)
    else:                                           # mid-run: NaN the six dropped after t0
        dff = df.copy()
        for j in dropped:
            dff, _, _ = inject(dff, ACTS[j], 'nan', a.t0, dur + 1)
        r = ThermalObserver(M).run(dff, a.dt)
    post = t >= a.t0
    worst = 0.0
    for j in dropped:
        e = (r['eff'][post, j] - true_all[post, j])
        ok = np.isfinite(true_all[post, j])
        RMSE[si, j] = np.sqrt(np.nanmean(e[ok] ** 2)); MAXE[si, j] = np.nanmax(np.abs(e[ok]))
        worst = max(worst, MAXE[si, j])
    mean_rmse = np.nanmean(RMSE[si])
    rows.append((S, mean_rmse, np.nanmax(MAXE[si]),
                 ACTS[int(np.nanargmax(np.where(np.isnan(MAXE[si]), -1, MAXE[si])))]))

# ---- heatmap of dropped-joint RMSE per survivor ----
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(RMSE, cmap='RdYlGn_r', vmin=0, vmax=20)
ax.set_xticks(range(NM)); ax.set_xticklabels(ACTS, rotation=45, ha='right')
ax.set_yticks(range(NM)); ax.set_yticklabels(ACTS)
ax.set_xlabel('dropped joint (predicted)'); ax.set_ylabel('the ONE surviving thermistor')
for i in range(NM):
    for j in range(NM):
        if np.isfinite(RMSE[i, j]):
            ax.text(j, i, '%.1f' % RMSE[i, j], ha='center', va='center', fontsize=8)
        else:
            ax.text(j, i, 'live', ha='center', va='center', fontsize=7, color='blue')
mode = 'boot' if a.t0 <= 0 else 'mid-run @%.0f min' % a.t0
ax.set_title('Virtual-prediction RMSE [C] — only 1 survivor, 6 dropped (%s, %s)'
             % (a.case.split('/')[-1], mode))
fig.colorbar(im, label='RMSE [C]  (capped at 20)')
tag = a.case.split('/')[-1].replace('.csv', '')
out = C.fig_path('dropout_solo_%s_%s.png' % (tag, 'boot' if a.t0 <= 0 else 'mid')); plt.tight_layout(); plt.savefig(out, dpi=600)

# ---- optional per-joint time series for a chosen survivor ----
if a.survivor:
    S = a.survivor; si = ACTS.index(S); dropped = [j for j in range(NM) if j != si]
    if a.t0 <= 0:
        r = ThermalObserver(M, forced_drop=dropped).run(df, a.dt)
    else:
        dff = df.copy()
        for j in dropped:
            dff, _, _ = inject(dff, ACTS[j], 'nan', a.t0, dur + 1)
        r = ThermalObserver(M).run(dff, a.dt)
    col = dict(zip(ACTS, plt.cm.tab10.colors))
    fig2, axes = plt.subplots(4, 2, figsize=(15, 16), sharex=True); axes = axes.ravel()
    for j, J in enumerate(ACTS):
        ax = axes[j]; true = true_all[:, j]; virt = r['eff'][:, j]
        ax.plot(t, true, lw=1.5, color=col[J], label='true (withheld)')
        if J == S:
            ax.set_title('%s  — SURVIVING thermistor (clamped to experiment)' % J, fontsize=10)
        else:
            ax.plot(t, virt, '--', lw=1.6, color='k', label='virtual (model)')
            e = virt - true; ok = np.isfinite(true)
            ax.set_title('%s  dropped   max err %.1f C   RMSE %.1f C' %
                         (J, np.nanmax(np.abs(e[ok])), np.sqrt(np.nanmean(e[ok] ** 2))), fontsize=10)
            if a.t0 > 0:
                ax.axvline(a.t0, color='r', ls=':', lw=1.2)
        ax.set_ylabel('motor T [C]'); ax.grid(alpha=.3); ax.legend(fontsize=8, loc='best')
    if len(ACTS) < len(axes):
        axes[-1].axis('off')
    for ax in axes[-2:]:
        ax.set_xlabel('time [min]')
    plt.suptitle('Only %s survives, other 6 predicted (%s, %s)' % (S, a.case.split('/')[-1], mode), fontsize=13)
    out2 = C.fig_path('dropout_solo_%s_%s_%s_ts.png' % (tag, 'boot' if a.t0 <= 0 else 'mid', S))
    plt.tight_layout(); plt.savefig(out2, dpi=600); print('saved', out2)

print('ONLY 1 survivor, 6 dropped (%s), %s, run %.0f min' % (a.case.split('/')[-1], mode, dur))
print('%-8s %12s %12s  %s' % ('survivor', 'mean RMSE', 'worst max', 'worst joint'))
for S, mr, mx, wj in sorted(rows, key=lambda z: z[1]):
    print('%-8s %11.1f  %11.1f  %s' % (S, mr, mx, wj))
print('\nsaved', out)
