"""Validate / fit the arm ROM against real-robot operation data (cleaned long format).

    python validate_robot.py --csv c_1261_arm_cleaned.csv --burst 1 --amb-offset -6          # compare only
    python validate_robot.py --csv c_1261_arm_cleaned.csv --burst 1 --amb-offset -6 --fit     # fit to that burst, then compare

Cleaned file columns: timestamp_utc, J, power_W, real_power_W, Tmotor_degC, Tamb_filled.
Data comes in BURSTS (long gaps between) so each burst is handled on its own.
With --fit, the conductances (R2, R_link, b, R_torso) are fit to the burst's measured
motor temps; result saved to fits/arm_robot_burst<N>.json (does NOT overwrite the main fit).

No structure TCs -> structures initialized by assumption:
  motors = measured Tmotor[0]; covered metal / shoulder = reference motor temp;
  covered fabric & bare metal = (reference motor + ambient)/2.   FET 4 W on, torso fixed.
Output: figures/<limb>_robot_burst<N>.png
"""
import argparse, json, numpy as np, pandas as pd
from scipy.optimize import least_squares
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import _common as C

ap = argparse.ArgumentParser()
ap.add_argument('--limb', default='arm')
ap.add_argument('--csv', required=True)
ap.add_argument('--burst', default='longest', help="burst index, 'longest', or 'all'")
ap.add_argument('--amb-offset', type=float, default=-6.0)
ap.add_argument('--gap', type=float, default=120.0)
ap.add_argument('--dt', type=float, default=4.0)
ap.add_argument('--fit', action='store_true', help='fit conductances to the (single) selected burst')
ap.add_argument('--skip-idle', action='store_true', help='start at sustained power onset (skip the no-power lead-in)')
ap.add_argument('--pthresh', type=float, default=0.05, help='1-min-avg total real power [W] that marks the sustained burst')
ap.add_argument('--start-min', type=float, default=None, help='manual start time [min] (overrides auto onset)')
ap.add_argument('--params', default=None, help='fit json to load (e.g. arm_robot_burst1); default = main fit')
a = ap.parse_args()

REF = {'shoulder': ['J1', 'J2'], 'hum_u': ['J2', 'TWIST'], 'hum_l': ['TWIST'],
       'fore_u': ['ELBOW'], 'fore_l': ['ROLL'], 'wrist': ['PITCH'], 'hand': ['YAW']}
M = C.rom(a.limb)
if a.params:
    import os
    M.build(np.array(json.load(open(os.path.join(C.FITS_DIR, a.params + '.json')))['p']))
else:
    M.build(C.load_params(a.limb))

raw = pd.read_csv(a.csv); raw['t'] = pd.to_datetime(raw['timestamp_utc']); raw = raw.sort_values('t')
ts = raw['t'].drop_duplicates().sort_values()
raw['burst'] = raw['t'].map(dict(zip(ts, (ts.diff().dt.total_seconds() > a.gap).cumsum())))
durs = raw.groupby('burst')['t'].agg(lambda s: (s.max() - s.min()).total_seconds())
if a.burst == 'longest':
    bursts = [int(durs.idxmax())]
elif a.burst == 'all':
    bursts = sorted(raw['burst'].unique())
else:
    bursts = [int(a.burst)]


def prep(b):
    """resample burst b to a uniform grid; return (df, x0, Tm dict, grid)."""
    d = raw[raw['burst'] == b]; t0 = d['t'].min(); d = d.assign(s=(d['t'] - t0).dt.total_seconds())
    grid = np.arange(0, d['s'].max() + 1e-6, a.dt)
    Tm = {}; P = {}
    for J in M.ACTS:
        g = d[d['J'] == J].sort_values('s')
        if g.empty:
            Tm[J] = None; P[J] = np.zeros_like(grid); continue
        Tm[J] = np.interp(grid, g['s'], g['Tmotor_degC'])
        P[J] = np.interp(grid, g['s'], g['real_power_W'])
    amb = np.interp(grid, d.sort_values('s')['s'], d.sort_values('s')['Tamb_filled']) + a.amb_offset
    on = None
    if a.start_min is not None:                      # manual start time
        on = int(np.argmax(grid >= a.start_min * 60))
    elif a.skip_idle:                                # auto: first SUSTAINED power (1-min rolling avg)
        ptot = np.sum([P[J] for J in M.ACTS], axis=0)
        w = max(1, int(round(60.0 / a.dt)))
        roll = pd.Series(ptot).rolling(w, min_periods=1, center=True).mean().to_numpy()
        cand = np.where(roll > a.pthresh)[0]
        on = int(cand[0]) if len(cand) else None
    if on:
        print('  (start at %.1f min, skipping %.1f min idle)' % (grid[on] / 60, grid[on] / 60))
        grid = grid[on:] - grid[on]; amb = amb[on:]
        for J in M.ACTS:
            P[J] = P[J][on:]
            if Tm[J] is not None:
                Tm[J] = Tm[J][on:]
    df = pd.DataFrame({'t_s': grid, 'T_amb': amb})
    for J in M.ACTS:
        df['P_' + J] = P[J]; df['Tm_' + J] = Tm[J] if Tm[J] is not None else np.nan
    amb0 = amb[0]
    reft = {s: np.nanmean([Tm[m][0] if Tm[m] is not None else np.nan for m in REF[s]]) for s in M.STRUCTS}
    x0 = np.zeros(M.NX)
    for i, J in enumerate(M.ACTS):
        x0[i] = Tm[J][0] if Tm[J] is not None else reft[M.TOPO[J][0]]
    for s in M.STRUCTS:
        x0[M.SIDX[s]] = reft[s] if (s in M.FABRICS or s in M.cfg.ENCLOSED) else (reft[s] + amb0) / 2
    for s in M.FABRICS:
        x0[M.FIDX[s]] = (reft[s] + amb0) / 2
    return df, x0, Tm, grid


def fit_to(b):
    df, x0, Tm, grid = prep(b)
    inst = [J for J in M.ACTS if Tm[J] is not None]

    def resid(p):
        M.build(p); X = M.simulate(df, a.dt, x0=x0)
        return np.concatenate([X[:, M.ACTS.index(J)] - Tm[J] for J in inst])
    L = np.log10; NM = M.NM
    p0 = np.concatenate([L([0.7] * NM), L([5] * NM), L([8.0] * M.NB)])
    lo = np.concatenate([L([0.05] * NM), L([0.05] * NM), L([1.0] * M.NB)])   # let b go to natural-convection
    hi = np.concatenate([L([1.4] * NM), L([15] * NM), L([40] * M.NB)])
    if M.has_torso:
        p0 = np.append(p0, L(1.5)); lo = np.append(lo, L(0.05)); hi = np.append(hi, L(5))
    print('fitting burst %d (%d steps, %d joints, %d params)...' % (b, len(grid), len(inst), len(p0)))
    sol = least_squares(resid, p0, bounds=(lo, hi), max_nfev=400)
    M.build(sol.x)
    R2, Rl, bb = M.unpack(sol.x)
    out = dict(R2=dict(zip(M.ACTS, R2.round(3))), R_link=dict(zip(M.ACTS, Rl.round(3))),
               b=dict(zip(M.BSTRUCTS, bb.round(2))), p=list(sol.x))
    if M.has_torso:
        out['R_torso'] = round(float(M.R_TORSO), 3)
    path = C.fit_path(a.limb).replace('_fitted_params', '_robot_burst%d' % b)
    json.dump(out, open(path, 'w'), indent=1, default=float)
    print('fit RMSE=%.2f C  ->  saved %s' % (np.sqrt(np.mean(sol.fun ** 2)), path))
    print('  b:', out['b']); print('  R_torso:', out.get('R_torso'))


def run_burst(b):
    df, x0, Tm, grid = prep(b)
    X = M.simulate(df, a.dt, x0=x0); tmin = grid / 60.0
    inst = [J for J in M.ACTS if Tm[J] is not None]; col = dict(zip(M.ACTS, plt.cm.tab10.colors))
    fig, ax = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    errs = []
    for i, J in enumerate(M.ACTS):
        if Tm[J] is None:
            continue
        ax[0].plot(tmin, Tm[J], lw=1.8, color=col[J], label=J)
        ax[0].plot(tmin, X[:, i], '--', lw=1.3, color=col[J])
        ax[1].plot(tmin, X[:, i] - Tm[J], lw=1.4, color=col[J], label=J)
        errs.append(X[:, i] - Tm[J])
    ax[0].plot(tmin, df['T_amb'], 'k:', lw=1.5, label='ambient')
    ax[0].set_ylabel('Tmotor [°C] (solid=meas, dashed=model)')
    ax[0].set_title('%s ROM vs robot burst %d (%.0f min, torso=%.0fC, ambient%+.0f)%s'
                    % (a.limb, b, tmin[-1], M.torso_temp, a.amb_offset, '  [FIT]' if a.fit else ''))
    ax[1].axhline(0, color='k', lw=.5); ax[1].set_ylabel('model − meas [°C]')
    for J in inst:
        ax[2].plot(tmin, df['P_' + J], lw=1.4, color=col[J], label=J)
    ax[2].set_ylabel('real copper power [W]'); ax[2].set_xlabel('time [min]')
    for x in ax:
        x.grid(alpha=.3); x.legend(fontsize=8, ncol=8, loc='upper left')
    out = C.fig_path('%s_robot_burst%d.png' % (a.limb, b)); plt.tight_layout(); plt.savefig(out, dpi=600); plt.close()
    errs = np.array(errs)
    print('burst %d (%.0f min): motor RMSE=%.2f C  mean|err|=%.2f C -> %s' %
          (b, tmin[-1], np.sqrt(np.mean(errs ** 2)), np.mean(np.abs(errs)), out))
    for i, J in enumerate(M.ACTS):
        if Tm[J] is not None:
            print('   %-6s model %4.1f  meas %4.1f  err %+5.1f' % (J, X[-1, i], Tm[J][-1], X[-1, i] - Tm[J][-1]))


if a.fit:
    if len(bursts) != 1:
        raise SystemExit('--fit needs a single --burst')
    fit_to(bursts[0])
for b in bursts:
    run_burst(b)
