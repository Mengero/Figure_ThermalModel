"""Open-loop field-prediction accuracy (experiment vs simulation) on robot UPS/depal data.

The model runs fully open-loop (every motor predicted from power + ambient + battery-torso,
no thermistor feedback) — the field-prediction accuracy and the ceiling for the dropout study.

  --csv <path>   per-case diagram: motors exp vs sim + error + per-actuator power
                 -> figures/<case>_predict.png
  --all          per-joint RMSE table across ALL standard cases + grouped bar chart
                 -> figures/field_compare.png

Examples:
  python studies/field_accuracy.py --csv data/depal/c_1092_depal_cleaned.csv --dt 4
  python studies/field_accuracy.py --all
"""
import argparse, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.join(os.path.dirname(HERE), 'pipeline'); sys.path.insert(0, PIPE)
import _common as C                                   # noqa: E402
from fields import load_field                          # noqa: E402
sys.path.insert(0, C.ROOT)
from core.observer import ThermalObserver              # noqa: E402

# standard field cases: label, path, dt, amb_offset
CASES = [
    ('c_1081 UPS',      'data/ups/c_1081_ups_cleaned.csv',   4.0, 0.0),
    ('c_1261 UPS',      'data/ups/c_1261_ups_cleaned.csv',   10.0, 0.0),
    ('0517 UPS(idle)',  'data/ups/ups_0517_idle.csv',        10.0, 0.0),
    ('c_1092 depal',    'data/depal/c_1092_depal_cleaned.csv', 4.0, 0.0),
    ('c_1076 depal',    'data/depal/c_1076_depal.csv',       10.0, 0.0),
]

ap = argparse.ArgumentParser()
ap.add_argument('--limb', default='arm')
ap.add_argument('--csv', default=None, help='single case: per-case prediction diagram')
ap.add_argument('--all', action='store_true', help='all standard cases: RMSE table + bar chart')
ap.add_argument('--dt', type=float, default=4.0)
ap.add_argument('--amb-offset', type=float, default=0.0)
a = ap.parse_args()

M = C.rom(a.limb); M.build(C.load_params(a.limb))


def predict(df, dt):
    """open-loop: force every motor dropped so eff = pure model prediction."""
    r = ThermalObserver(M, forced_drop=tuple(range(M.NM))).run(df, dt)
    meas = r['meas'].copy(); meas[(meas < 0) | (meas > 150)] = np.nan
    return r['t'], r['eff'], meas


# ---------------- single case ----------------
if a.csv:
    df, _ = load_field(M, a.csv, a.dt, a.amb_offset)
    tag = os.path.splitext(os.path.basename(a.csv))[0]
    t, pred, meas = predict(df, a.dt)
    order = M.ACTS; col = dict(zip(order, plt.cm.tab10.colors))
    fig, ax = plt.subplots(3, 1, figsize=(13, 14), sharex=True)
    rmse = {}
    for i, J in enumerate(order):
        m = meas[:, i]
        if not np.isfinite(m).any():
            continue
        rmse[J] = np.sqrt(np.nanmean((pred[:, i] - m) ** 2))
        ax[0].plot(t, m, lw=1.4, color=col[J], label='%s (RMSE %.1f)' % (J, rmse[J]))
        ax[0].plot(t, pred[:, i], '--', lw=1.1, color=col[J])
        ax[1].plot(t, pred[:, i] - m, lw=1.0, color=col[J], label=J)
        ax[2].plot(t, df['P_' + J], lw=1.0, color=col[J], label=J)
    if 'T_torso' in df:
        ax[0].plot(t, df['T_torso'], 'k:', lw=1.2, label='battery/torso')
    ax[0].plot(t, df['T_amb'], color='gray', ls=':', lw=1.2, label='ambient')
    ax[1].axhline(0, color='k', lw=.5)
    cols = [i for i in range(M.NM) if np.isfinite(meas[:, i]).any()]
    allr = np.sqrt(np.nanmean((pred - meas)[:, cols] ** 2))
    ax[0].set_ylabel('motor T [C] (solid=exp, dashed=sim)')
    ax[0].set_title('%s — open-loop prediction vs experiment (overall motor RMSE %.2f C)' % (tag, allr))
    ax[0].grid(alpha=.3); ax[0].legend(fontsize=8, ncol=4, loc='upper left')
    ax[1].set_ylabel('sim - exp [C]'); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8, ncol=7)
    ax[2].set_ylabel('motor copper power [W]'); ax[2].set_xlabel('time [min]'); ax[2].grid(alpha=.3); ax[2].legend(fontsize=8, ncol=7)
    short = '_'.join(tag.split('_')[:3]).lower()
    out = C.fig_path('field_%s.png' % short, a.limb); plt.tight_layout(); plt.savefig(out, dpi=600)
    print('overall motor RMSE = %.2f C  ->  saved %s' % (allr, out))

# ---------------- all cases ----------------
elif a.all:
    res = {}
    print('%-16s %6s  %s' % ('case', 'dur', '  '.join('%5s' % aa for aa in M.ACTS) + '   ALL'))
    for label, path, dt, off in CASES:
        df, dur = load_field(M, path, dt, off)
        t, pred, meas = predict(df, dt)
        rm = [np.sqrt(np.nanmean((pred[:, i] - meas[:, i]) ** 2)) if np.isfinite(meas[:, i]).any() else np.nan
              for i in range(M.NM)]
        allr = np.sqrt(np.nanmean((pred - meas) ** 2))
        res[label] = rm
        print('%-16s %5.0fm  %s  %5.1f' % (label, dur, '  '.join(('%5.1f' % v if np.isfinite(v) else '  -- ') for v in rm), allr))
    fig, ax = plt.subplots(figsize=(13, 6)); x = np.arange(M.NM); w = 0.16
    for n, (label, rm) in enumerate(res.items()):
        ax.bar(x + n * w, [v if np.isfinite(v) else 0 for v in rm], w, label=label)
    ax.axhline(15, color='k', ls='--', lw=1, label='+/-15 C')
    ax.set_xticks(x + 2 * w); ax.set_xticklabels(M.ACTS); ax.set_ylabel('open-loop prediction RMSE [C]')
    ax.set_title('Field prediction accuracy per joint — UPS vs depal'); ax.grid(axis='y', alpha=.3); ax.legend(fontsize=8)
    out = C.fig_path('field_compare.png', a.limb); plt.tight_layout(); plt.savefig(out, dpi=600); print('\nsaved', out)
else:
    ap.error('give --csv <case> or --all')
