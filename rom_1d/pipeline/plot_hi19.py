"""Blind validation on the arm's 3-joint hi19 endurance runs (J1/TWIST/ELBOW powered).

    python plot_hi19.py --limb arm

Only the instrumented motors + the J1 boundary are shown vs model; cold joints are
skipped. Dotted line = ~120 C derate clamp. Output: figures/<limb>_hi19_validation.png
This validation set is arm-specific (tags listed below); other limbs can reuse it
once they have analogous endurance runs.
"""
import argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import _common as C

ap = argparse.ArgumentParser()
ap.add_argument('--limb', default='arm')
ap.add_argument('--tags', nargs='+', default=['hi19a', 'hi19b'])
a = ap.parse_args()

M = C.rom(a.limb)
M.build(C.load_params(a.limb))
covered, bare = M.FABRICS, M.BARE
palette = plt.cm.tab10.colors
COL = {s: palette[i % 10] for i, s in enumerate(M.STRUCTS)}

fig, axes = plt.subplots(4, len(a.tags), figsize=(8 * len(a.tags), 18), squeeze=False)
for c, tag in enumerate(a.tags):
    df = C.load(a.limb, tag); X = M.simulate(df, 2.0); obs = M.struct_obs(X, df)
    t = df['t_s'] / 60
    ax = axes[0][c]; H = []; L = []
    for i, act in enumerate(M.ACTS):
        m = df['Tm_' + act].to_numpy()
        if not np.isfinite(m).any():
            continue
        ln, = ax.plot(t, m, lw=2.0, alpha=.9)
        ax.plot(t, X[:, i], '--', color=ln.get_color(), lw=1.5); H.append(ln); L.append(act)
    ax.axhline(120, color='k', ls=':', lw=1.2, alpha=.7)
    ax.legend(H, L, fontsize=9, loc='upper right')
    ax.set_title(f'{tag} — MOTORS (solid=exp, dashed=model; dotted=120C clamp)')
    ax.grid(axis='x', alpha=.3); ax.set_ylabel('T [°C]')
    for r, grp, name in [(1, covered, 'COVERED STRUCTURES'), (2, bare, 'BARE STRUCTURES')]:
        ax = axes[r][c]; H = []; L = []
        for s in grp:
            ln, = ax.plot(t, df[M.TS_COLS[s]], lw=1.8, alpha=.9, color=COL[s])
            ax.plot(t, obs[s], '--', color=COL[s], lw=1.4); H.append(ln); L.append(s)
        ax.legend(H, L, fontsize=9, loc='upper right')
        ax.set_title(f'{tag} — {name} (solid=TC, dashed=model)'); ax.grid(axis='x', alpha=.3); ax.set_ylabel('T [°C]')
    ax = axes[3][c]; H = []
    for act in M.ACTS:
        ln, = ax.plot(t, df['P_' + act], lw=1.5); H.append(ln)
    ax.legend(H, M.ACTS, fontsize=8, ncol=4, loc='upper right')
    ax.set_title(f'{tag} — MOTOR POWER'); ax.grid(axis='x', alpha=.3); ax.set_xlabel('time [min]'); ax.set_ylabel('P [W]')

out = C.fig_path('hi19_validation.png', a.limb)
plt.tight_layout(); plt.savefig(out, dpi=600)
print(f'saved {out}')
