"""Train- or test-fit overview for one limb (solid = experiment, dashed = model).

    python plot_fit.py --limb arm --split train
    python plot_fit.py --limb arm --split test

Rows (config-driven, so any limb works):
  motors / covered structures / bare structures / simulated metal nodes /
  dP/dt per motor / adaptive motor C(t) / motor power.
Also prints a per-case RMSE summary. Output: figures/<limb>_<split>_fit.png
"""
import argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import _common as C

ap = argparse.ArgumentParser()
ap.add_argument('--limb', default='arm')
ap.add_argument('--split', choices=['train', 'test'], default='train')
a = ap.parse_args()

M = C.rom(a.limb)
M.build(C.load_params(a.limb))
covered, bare = M.FABRICS, M.BARE
metal_nodes = list(M.cfg.ENCLOSED) + covered + bare        # all simulated metal structures
palette = plt.cm.tab10.colors
COL = {s: palette[i % 10] for i, s in enumerate(M.STRUCTS)}
tags = M.cfg.TRAIN if a.split == 'train' else M.cfg.TEST

fig, axes = plt.subplots(7, len(tags), figsize=(8 * len(tags), 27), squeeze=False)
for c, tag in enumerate(tags):
    df = C.load(a.limb, tag)
    X = M.simulate(df, 2.0); obs = M.struct_obs(X, df)
    t = df['t_s'] / 60

    ax = axes[0][c]; H = []; L = []
    for i, act in enumerate(M.ACTS):
        m = df['Tm_' + act].to_numpy()
        if not np.isfinite(m).any():
            continue
        ln, = ax.plot(t, m, lw=1.8, alpha=.9)
        ax.plot(t, X[:, i], '--', color=ln.get_color(), lw=1.3); H.append(ln); L.append(act)
    ax.legend(H, L, fontsize=8, ncol=2, loc='upper left')
    ax.set_title(f'{tag} — MOTORS (solid=exp, dashed=model)'); ax.grid(alpha=.3)
    if c == 0: ax.set_ylabel('T [°C]')

    for r, grp, name in [(1, covered, 'COVERED STRUCTURES'), (2, bare, 'BARE STRUCTURES')]:
        ax = axes[r][c]; H = []; L = []
        for s in grp:
            if s not in M.TS_COLS:                        # no TC (e.g. leg foot): nothing to compare, skip
                continue
            ln, = ax.plot(t, df[M.TS_COLS[s]], lw=1.8, alpha=.9, color=COL[s])
            ax.plot(t, obs[s], '--', color=COL[s], lw=1.3); H.append(ln); L.append(s)
        ax.legend(H, L, fontsize=8, ncol=2, loc='upper left')
        ax.set_title(f'{tag} — {name}'); ax.grid(alpha=.3)
        if c == 0: ax.set_ylabel('T [°C]')

    ax = axes[3][c]; H = []; L = []
    for s in metal_nodes:
        ln, = ax.plot(t, X[:, M.SIDX[s]], lw=1.6, alpha=.9, color=COL[s]); H.append(ln); L.append(s)
    ax.legend(H, L, fontsize=8, ncol=2, loc='upper left')
    ax.set_title(f'{tag} — METAL STRUCTURES (simulated)'); ax.grid(alpha=.3)
    if c == 0: ax.set_ylabel('T [°C]')

    ax = axes[4][c]; H = []; ts = df['t_s'].to_numpy()
    for act in M.ACTS:
        ln, = ax.plot(t, np.gradient(df['P_' + act].to_numpy(), ts), lw=1.2); H.append(ln)
    ax.axhline(0, color='k', lw=.5); ax.legend(H, M.ACTS, fontsize=7, ncol=2, loc='upper right')
    ax.set_title(f'{tag} — dP/dt [W/s]'); ax.grid(alpha=.3)
    if c == 0: ax.set_ylabel('dP/dt [W/s]')

    ax = axes[5][c]; H = []; L = []; Cser = M.cmotor_series(df)
    for i in M.free_mot:
        ln, = ax.plot(t, Cser[:, i], lw=1.4); H.append(ln); L.append(M.ACTS[i])
    ax.legend(H, L, fontsize=7, ncol=2, loc='lower right')
    ax.set_title(f'{tag} — motor C(t) [J/K] (winding↔assembly)'); ax.grid(alpha=.3)
    if c == 0: ax.set_ylabel('C [J/K]')

    ax = axes[6][c]; H = []
    for act in M.ACTS:
        ln, = ax.plot(t, df['P_' + act], lw=1.5); H.append(ln)
    ax.legend(H, M.ACTS, fontsize=7, ncol=2, loc='upper right')
    ax.set_title(f'{tag} — MOTOR POWER'); ax.grid(alpha=.3); ax.set_xlabel('time [min]')
    if c == 0: ax.set_ylabel('P [W]')

out = C.fig_path(f'{a.limb}_{a.split}_fit.png')
plt.tight_layout(); plt.savefig(out, dpi=600)
print(f'saved {out}')

# ---- per-case RMSE summary ----
for tag in tags:
    df = C.load(a.limb, tag); X = M.simulate(df, 2.0); obs = M.struct_obs(X, df)
    print(f"--- {tag} ---")
    for i, act in enumerate(M.ACTS):
        if i in M.boundary_idx:
            print(f"  Tm_{act:6s} (measured boundary — not predicted)"); continue
        m = df['Tm_' + act].to_numpy()
        if not np.isfinite(m).any():
            continue
        ok = np.isfinite(m) & (m < 118.0)                 # mask ~120 C derate clamp
        if ok.sum() == 0:
            continue
        endi = np.where(ok)[0][-1]
        print(f"  Tm_{act:6s} RMSE={np.sqrt(np.mean((X[ok,i]-m[ok])**2)):6.2f}"
              f"  end model={X[endi,i]:6.1f} exp={m[endi]:6.1f}")
    for s, col in M.TS_COLS.items():
        e = obs[s] - df[col].to_numpy()
        print(f"  Ts_{s:6s} RMSE={np.sqrt(np.mean(e**2)):6.2f}")
