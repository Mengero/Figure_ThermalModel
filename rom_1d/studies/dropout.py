"""Thermistor-dropout studies — virtual-prediction accuracy when motor sensors drop out.

Modes (--mode):
  midrun   drop ONE joint at --t0 minutes (sweep all 7); score the post-dropout window.
  boot     drop ONE joint from t=0 (sweep all 7); score the whole run.
  solo     ONLY ONE thermistor survives, the other six drop from boot; sweep the survivor
           -> RMSE heatmap. With --survivor J also plot that scenario's per-joint time series.
  allzero  ALL thermistors drop; the model runs open-loop, seeded from the battery temp.

Examples:
  python studies/dropout.py --mode midrun  --case data/depal/c_1092_depal_cleaned.csv --dt 4 --t0 82
  python studies/dropout.py --mode boot    --case data/ups/c_1081_ups_cleaned.csv     --dt 4
  python studies/dropout.py --mode solo    --case data/depal/c_1092_depal_cleaned.csv --dt 4 --survivor J2
  python studies/dropout.py --mode allzero --case data/depal/c_1092_depal_cleaned.csv --dt 4

Figures land in figures/dropout_<mode>_<case>[...].png
"""
import argparse, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.join(os.path.dirname(HERE), 'pipeline'); sys.path.insert(0, PIPE)
import _common as C                                   # noqa: E402
from fields import load_field                          # noqa: E402
import plots as PL                                     # noqa: E402
sys.path.insert(0, C.ROOT)
from core.observer import ThermalObserver              # noqa: E402
from _faults import inject                             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--mode', required=True, choices=['midrun', 'boot', 'solo', 'allzero'])
ap.add_argument('--case', default='data/depal/c_1092_depal_cleaned.csv')
ap.add_argument('--dt', type=float, default=4.0)
ap.add_argument('--t0', type=float, default=82.0, help='dropout instant [min] (midrun)')
ap.add_argument('--survivor', default=None, help='also plot this survivor scenario (solo)')
a = ap.parse_args()

M = C.rom('arm'); M.build(C.load_params('arm'))
df, dur = load_field(M, a.case, a.dt, 0.0)
t = df['t_s'].to_numpy() / 60.0
ACTS = M.ACTS; NM = M.NM
col = dict(zip(ACTS, plt.cm.tab10.colors))
true_all = df[['Tm_' + j for j in ACTS]].to_numpy(dtype=float)
true_all[(true_all < 0) | (true_all > 150)] = np.nan      # mask dead-sensor garbage
Tbatt = df['T_torso'].to_numpy() if 'T_torso' in df else np.full(len(df), M.torso_temp)
tag = os.path.basename(a.case).replace('.csv', '')


def run(forced=None, inject_nan=None):
    """One observer run. forced: list of dropped indices (boot). inject_nan: (idxs, t0) mid-run."""
    if inject_nan is not None:
        idxs, t0 = inject_nan; d = df.copy()
        for j in idxs:
            d, _, _ = inject(d, ACTS[j], 'nan', t0, dur + 1)
        return ThermalObserver(M).run(d, a.dt)
    return ThermalObserver(M, forced_drop=forced or []).run(df, a.dt)


# ---- per-joint sweep (midrun / boot): drop each joint alone, one panel each ----
if a.mode in ('midrun', 'boot'):
    post = t >= a.t0 if a.mode == 'midrun' else None
    fig, axes = PL.motor_grid()
    print('%s dropout (%s), run %.0f min' % (a.mode, tag, dur))
    print('%-6s %10s %10s %10s' % ('joint', 'max|err|', 'RMSE', 'end err'))
    for j, J in enumerate(ACTS):
        r = run(inject_nan=([j], a.t0)) if a.mode == 'midrun' else run(forced=[j])
        virt = r['eff'][:, j]; true = true_all[:, j]
        mx, rm, en = PL.score(virt, true, post)
        print('%-6s %9.1f  %9.1f  %+9.1f' % (J, mx, rm, en))
        ttl = ('%s @%.0f min  max %.1f  RMSE %.1f' % (J, a.t0, mx, rm) if a.mode == 'midrun'
               else '%s @boot  max %.1f  RMSE %.1f  end %+.1f' % (J, mx, rm, en))
        PL.plot_true_vs_virtual(axes[j], t, true, virt, col[J], ttl, post=post)
        if a.mode == 'midrun':
            axes[j].axvline(a.t0, color='r', ls=':', lw=1.2)
    out = C.fig_path('dropout_%s_%s.png' % (a.mode, tag), 'studies')
    PL.finish_grid(fig, axes, NM, '%s single-sensor dropout vs withheld truth (%s)' % (a.mode, tag), out)

# ---- all dropped: one open-loop run, seeded from battery ----
elif a.mode == 'allzero':
    r = run(forced=list(range(NM)))
    fig, axes = PL.motor_grid()
    print('all dropped, seeded from battery %.1f C (%s), run %.0f min' % (Tbatt[0], tag, dur))
    print('%-6s %10s %10s %10s' % ('joint', 'max|err|', 'RMSE', 'end err'))
    rmses = []
    for j, J in enumerate(ACTS):
        virt = r['eff'][:, j]; true = true_all[:, j]
        mx, rm, en = PL.score(virt, true)
        rmses.append(rm)
        print('%-6s %9s  %9s  %9s' % (J, '--' if np.isnan(mx) else '%.1f' % mx,
                                      '--' if np.isnan(rm) else '%.1f' % rm,
                                      '--' if np.isnan(en) else '%+.1f' % en))
        ttl = '%s   max %s   RMSE %s' % (J, '--' if np.isnan(mx) else '%.1f' % mx,
                                         '--' if np.isnan(rm) else '%.1f' % rm)
        PL.plot_true_vs_virtual(axes[j], t, true, virt, col[J], ttl, extra=('battery/torso', Tbatt))
    out = C.fig_path('dropout_allzero_%s.png' % tag, 'studies')
    PL.finish_grid(fig, axes, NM, 'ALL dropped — open-loop (init from battery %.1f C) — %s  mean RMSE %.1f C'
                   % (Tbatt[0], tag, np.nanmean(rmses)), out)

# ---- solo survivor: RMSE heatmap (+ optional per-joint time series) ----
elif a.mode == 'solo':
    RMSE = np.full((NM, NM), np.nan); MAXE = np.full((NM, NM), np.nan)
    rows = []
    for si, S in enumerate(ACTS):
        dropped = [j for j in range(NM) if j != si]
        r = run(forced=dropped)
        for j in dropped:
            MAXE[si, j], RMSE[si, j], _ = PL.score(r['eff'][:, j], true_all[:, j])
        rows.append((S, np.nanmean(RMSE[si]), np.nanmax(MAXE[si]),
                     ACTS[int(np.nanargmax(np.where(np.isnan(MAXE[si]), -1, MAXE[si])))]))
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(RMSE, cmap='RdYlGn_r', vmin=0, vmax=20)
    ax.set_xticks(range(NM)); ax.set_xticklabels(ACTS, rotation=45, ha='right')
    ax.set_yticks(range(NM)); ax.set_yticklabels(ACTS)
    ax.set_xlabel('dropped joint (predicted)'); ax.set_ylabel('the ONE surviving thermistor')
    for i in range(NM):
        for j in range(NM):
            ax.text(j, i, 'live' if np.isnan(RMSE[i, j]) else '%.1f' % RMSE[i, j],
                    ha='center', va='center', fontsize=8, color='blue' if np.isnan(RMSE[i, j]) else 'black')
    ax.set_title('Virtual-prediction RMSE [C] — only 1 survivor, 6 dropped (%s)' % tag)
    fig.colorbar(im, label='RMSE [C] (capped 20)')
    out = C.fig_path('dropout_solo_%s.png' % tag, 'studies'); fig.tight_layout(); fig.savefig(out, dpi=600)
    print('only 1 survivor, 6 dropped (%s), run %.0f min' % (tag, dur))
    print('%-8s %12s %12s  %s' % ('survivor', 'mean RMSE', 'worst max', 'worst joint'))
    for S, mr, mx, wj in sorted(rows, key=lambda z: z[1]):
        print('%-8s %11.1f  %11.1f  %s' % (S, mr, mx, wj))
    print('saved', out)

    if a.survivor:                                   # per-joint time series for one survivor
        S = a.survivor; si = ACTS.index(S)
        r = run(forced=[j for j in range(NM) if j != si])
        fig2, axes = PL.motor_grid()
        for j, J in enumerate(ACTS):
            true = true_all[:, j]; virt = r['eff'][:, j]
            if J == S:
                axes[j].plot(t, true, lw=1.5, color=col[J], label='true (clamped)')
                axes[j].set_title('%s — SURVIVING thermistor' % J, fontsize=9.5)
                axes[j].set_ylabel('motor T [C]'); axes[j].grid(alpha=.3); axes[j].legend(fontsize=8)
            else:
                mx, rm, _ = PL.score(virt, true)
                PL.plot_true_vs_virtual(axes[j], t, true, virt, col[J],
                                        '%s dropped  max %.1f  RMSE %.1f' % (J, mx, rm))
        out2 = C.fig_path('dropout_solo_%s_%s_ts.png' % (tag, S), 'studies')
        PL.finish_grid(fig2, axes, NM, 'Only %s survives, other 6 predicted (%s)' % (S, tag), out2)
        print('saved', out2)
