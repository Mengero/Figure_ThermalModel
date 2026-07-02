"""Plot the steady-state operating-point fit: model vs measured motor temps.

    python plot_steady.py --limb leg --steady steady_c1081

Uses the saved fit (fits/<limb>_fitted_params.json) and the steady case
(data/<limb>/<steady>.json). Solves the self-consistent steady state (copper
power depends on winding temp) exactly as fit.py's steady_residual does, and
plots measured vs model motor temp per actuator. Output: figures/<limb>_<steady>_steady.png
"""
import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt
import _common as C

ap = argparse.ArgumentParser()
ap.add_argument('--limb', default='leg')
ap.add_argument('--steady', default='steady_c1081')
a = ap.parse_args()

M = C.rom(a.limb)
M.build(C.load_params(a.limb))
d = json.load(open(os.path.join(C.DATA_DIR, a.limb, a.steady + '.json')))

ACTS = M.ACTS
iq = np.array([d['iq'].get(x, 0.0) for x in ACTS])
R20 = np.array([M.cfg.R20[x] for x in ACTS])
meas = np.array([d['Tm'].get(x, np.nan) for x in ACTS])
Tamb = d['T_amb']

if M.has_torso and 'T_torso' in d:
    M.torso_temp = d['T_torso']  # field case: torso = battery cell temp (else cfg fallback 40 C)
# self-consistent steady solve (matches engine.steady_residual)
Tw = np.where(np.isfinite(meas), meas, 80.0)
for _ in range(60):
    P = 1.5 * iq ** 2 * R20 * (234.5 + Tw) / (234.5 + 20.0)
    T = M.steady_state(P, Tamb)
    Tw = T[:M.NM]
model = T[:M.NM]

ok = np.isfinite(meas)
rmse = np.sqrt(np.mean((model[ok] - meas[ok]) ** 2))

x = np.arange(len(ACTS)); w = 0.38
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - w / 2, np.where(ok, meas, np.nan), w, label='measured', color='#1f77b4')
ax.bar(x + w / 2, model, w, label='model', color='#ff7f0e', alpha=.85)
for i in range(len(ACTS)):
    if ok[i]:
        ax.annotate(f'{model[i]-meas[i]:+.1f}', (x[i], max(model[i], meas[i]) + 0.5),
                    ha='center', fontsize=8, color='dimgray')
    else:
        ax.annotate('excluded', (x[i], model[i] + 0.5), ha='center', fontsize=7, color='red')
ax.axhline(Tamb, ls='--', lw=1, color='gray', label=f'ambient {Tamb:.1f} C')
if M.has_torso:
    ax.axhline(M.torso_temp, ls=':', lw=1, color='green', label=f'torso {M.torso_temp:.0f} C')
ax.set_xticks(x); ax.set_xticklabels(ACTS, rotation=30, ha='right')
ax.set_ylabel('motor temp [°C]')
ax.set_title(f'{a.limb} steady-state fit — {a.steady}  (RMSE={rmse:.2f} °C; +/- = model−measured)')
ax.legend(fontsize=8); ax.grid(axis='y', alpha=.3)
out = C.fig_path(f'{a.steady}_steady.png', a.limb)
plt.tight_layout(); plt.savefig(out, dpi=600)
print('wrote', out, ' steady RMSE=%.2f C' % rmse)
for i, aa in enumerate(ACTS):
    tag = 'excluded' if not ok[i] else f'meas={meas[i]:.1f}  model={model[i]:.1f}  err={model[i]-meas[i]:+.1f}'
    print(f'  {aa:11} {tag}')
