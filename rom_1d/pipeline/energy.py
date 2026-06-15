"""Energy audit for one limb (consolidates the old balance/dist/time scripts).

    python energy.py --limb arm            # all three views
    python energy.py --limb arm --mode dist

Views:
  balance — per-case closure check (generated + boundary == to-air + stored + residual)
  dist    — stacked-bar share of heat input per case (figures/<limb>_energy_dist.png)
  time    — time-resolved flow rates per case   (figures/<limb>_energy_time.png)

Motor storage uses the time-varying adaptive C (integral C(t) dT, not C_final*dT),
so the books close to ~100%.
"""
import argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import _common as C

ap = argparse.ArgumentParser()
ap.add_argument('--limb', default='arm')
ap.add_argument('--mode', choices=['all', 'balance', 'dist', 'time'], default='all')
a = ap.parse_args()

M = C.rom(a.limb)
M.build(C.load_params(a.limb))
CASES = list(M.cfg.TRAIN) + list(M.cfg.TEST)


def integrated(tag):
    """exact integrated energy breakdown [kJ] for one case."""
    df = C.load(a.limb, tag); X = M.simulate(df, 2.0)
    t = df['t_s'].to_numpy()
    fl = M.energy_flows(X, df)
    E = lambda q: np.trapezoid(q, t) / 1000.0
    gen = E(fl['gen']); ebnd = E(fl['bnd']); eamb = E(fl['amb'])
    Cmt = M.cmotor_series(df)
    dmot = sum(np.sum(Cmt[:-1, i] * np.diff(X[:, i])) for i in M.free_mot) / 1000.0
    dstr = (sum(M.cfg.C_S_FIX[s] * (X[-1, M.SIDX[s]] - X[0, M.SIDX[s]]) for s in M.STRUCTS) +
            sum(M.cfg.C_F_FIX[s] * (X[-1, M.FIDX[s]] - X[0, M.FIDX[s]]) for s in M.FABRICS)) / 1000.0
    inb, outb = max(ebnd, 0.0), max(-ebnd, 0.0)
    tot = gen + inb
    resid = tot - (dmot + dstr + eamb + outb)
    return dict(tag=tag, dur=t[-1] / 60, gen=gen, inb=inb, outb=outb,
                dmot=dmot, dstr=dstr, amb=eamb, resid=resid, tot=tot)


if a.mode in ('all', 'balance', 'dist'):
    rows = [integrated(tag) for tag in CASES]
    for r in rows:
        tot = r['tot']
        print(f"=== {r['tag']} ({r['dur']:.1f} min) ===  [kJ]  total input={tot:.1f}")
        print(f"  generated(in-model)     : {r['gen']:7.1f}")
        print(f"  from boundary (in)      : {r['inb']:7.1f}")
        print(f"  -> stored in MOTORS     : {r['dmot']:7.1f}  ({100*r['dmot']/tot:5.1f}%)")
        print(f"  -> stored in STRUCTURES : {r['dstr']:7.1f}  ({100*r['dstr']/tot:5.1f}%)")
        print(f"  -> to AMBIENT           : {r['amb']:7.1f}  ({100*r['amb']/tot:5.1f}%)")
        print(f"  -> to boundary (out)    : {r['outb']:7.1f}  ({100*r['outb']/tot:5.1f}%)")
        print(f"  -> residual             : {r['resid']:7.1f}  ({100*r['resid']/tot:5.1f}%)")

if a.mode in ('all', 'dist'):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    tags = [r['tag'] for r in rows]; tot = [r['tot'] for r in rows]
    cats = [('stored: motors', 'dmot', 'tab:red'), ('stored: structures', 'dstr', 'tab:orange'),
            ('to ambient', 'amb', 'tab:blue'), ('to boundary', 'outb', 'tab:purple'),
            ('residual', 'resid', 'lightgray')]
    bottom = np.zeros(len(rows))
    for lab, key, col in cats:
        vals = np.array([100 * r[key] / tt for r, tt in zip(rows, tot)])
        ax.bar(tags, vals, bottom=bottom, label=lab, color=col)
        for i, (v, bm) in enumerate(zip(vals, bottom)):
            if v > 4:
                ax.text(i, bm + v / 2, f"{v:.0f}%", ha='center', va='center', fontsize=9, color='white')
        bottom += vals
    ax.set_ylabel('share of heat input [%]')
    ax.set_title(f'{a.limb} — energy distribution per case'); ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=.3)
    out = C.fig_path(f'{a.limb}_energy_dist.png'); plt.tight_layout(); plt.savefig(out, dpi=600)
    print(f'saved {out}')

if a.mode in ('all', 'time'):
    nc = 3; nr = int(np.ceil(len(CASES) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(6.3 * nc, 5 * nr), squeeze=False)
    for k, tag in enumerate(CASES):
        ax = axes[k // nc][k % nc]
        df = C.load(a.limb, tag); X = M.simulate(df, 2.0); fl = M.energy_flows(X, df)
        ax.plot(fl['t'], fl['gen'], 'k', lw=2, label='generated')
        ax.plot(fl['t'], fl['bnd'], color='tab:purple', lw=1.5, label='from boundary')
        ax.plot(fl['t'], fl['mot'], color='tab:red', lw=1.7, label='stored: motors')
        ax.plot(fl['t'], fl['struct'], color='tab:orange', lw=1.7, label='stored: structures')
        ax.plot(fl['t'], fl['amb'], color='tab:blue', lw=1.7, label='to ambient')
        ax.set_title(tag); ax.grid(alpha=.3); ax.axhline(0, color='k', lw=.5); ax.set_xlabel('time [min]')
        if k % nc == 0: ax.set_ylabel('power [W]')
        if k == 0: ax.legend(fontsize=8)
    for k in range(len(CASES), nr * nc):
        axes[k // nc][k % nc].axis('off')
    plt.suptitle(f'{a.limb} — time-resolved energy flows', fontsize=14)
    out = C.fig_path(f'{a.limb}_energy_time.png'); plt.tight_layout(); plt.savefig(out, dpi=600)
    print(f'saved {out}')
