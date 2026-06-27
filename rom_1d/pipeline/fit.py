"""Grey-box fit of R2, R_link, b for one limb.

    python fit.py --limb arm
    python fit.py --limb arm --steady steady_061526 --steady-weight 10

Writes fits/<limb>_fitted_params.json. Capacitances are fixed (not fit); the
adaptive motor C is handled inside the engine. With --steady, a steady-state
operating point (data/<limb>/<case>.json: iq + measured motor temps + ambient)
is folded into the fit as an extra weighted residual block — the only data that
constrains the torso sink and the high-power distal convection.
"""
import argparse, time, json, os
import numpy as np
from scipy.optimize import least_squares
import _common as C

ap = argparse.ArgumentParser()
ap.add_argument('--limb', default='arm')
ap.add_argument('--steady', default='steady_061526',
                help="steady-state case to fold into training (data/<limb>/<case>.json); '' to disable")
ap.add_argument('--steady-weight', type=float, default=10.0,
                help="per-residual weight on the steady point (it's 7 pts vs ~thousands transient)")
ap.add_argument('--fix-r2', action='store_true',
                help="fix R2 to size-based values (SM85=1.96, SM72=1.0, SM44L=1.0475 K/W); fit only R_link/b/(R_torso)")
a = ap.parse_args()
R2_BY_SIZE = {'SM85': 1.96, 'SM72': 1.0, 'SM44L': 1.0475}   # K/W: SM72 from J1 fit, SM85 hip_y meas, SM44L avg(PITCH,YAW)

M = C.rom(a.limb)
runs = [C.load(a.limb, t) for t in M.cfg.TRAIN]

# optional steady-state operating point folded into training
SP = None
sp_path = os.path.join(C.DATA_DIR, a.limb, str(a.steady) + '.json')
if a.steady and M.cfg.R20 and os.path.exists(sp_path):
    d = json.load(open(sp_path))
    SP = dict(iq=np.array([d['iq'].get(x, 0.0) for x in M.ACTS]),
              R20=np.array([M.cfg.R20[x] for x in M.ACTS]),
              meas=np.array([d['Tm'].get(x, np.nan) for x in M.ACTS]),
              Tamb=d['T_amb'], W=a.steady_weight, tag=a.steady)


def resid(q):
    p = expand(q)                                              # splice in fixed R2 if --fix-r2
    rt = M.residuals(p, runs)                                   # transient (builds for p)
    if SP is None:
        return rt
    rs = M.steady_residual(SP['iq'], SP['R20'], SP['Tamb'], SP['meas']) * SP['W']
    return np.concatenate([rt, rs])
L = np.log10
NM = M.NM
if 'J1' in M.ACTS:                            # arm: cap J1 R2 tighter; others 1.4 K/W
    r2_hi = [1.4] * NM
    r2_hi[M.ACTS.index('J1')] = 1.0
else:                                         # leg: SM85 R2 (~1.96 K/W) exceeds the arm's 1.4 cap
    r2_hi = [3.0] * NM
rest_p0 = np.concatenate([L([5] * NM), L([20] * M.NB)])        # R_link, b (always fitted)
rest_lo = np.concatenate([L([0.0001] * NM), L([3] * M.NB)])
rest_hi = np.concatenate([L([15] * NM), L([40] * M.NB)])
if a.fix_r2 or M.cfg.FIX_R2:                   # R2 NOT optimized: solve only R_link/b/(R_torso)
    r2fix_log = L(np.array([R2_BY_SIZE[M.cfg.MOTOR[x]] for x in M.ACTS]))
    expand = lambda q: np.concatenate([r2fix_log, q])         # prepend fixed R2 -> full p
    p0, lo, hi = rest_p0.copy(), rest_lo.copy(), rest_hi.copy()
else:                                          # R2 free: full vector [R2, R_link, b]
    expand = lambda q: q
    p0 = np.concatenate([L([0.7] * NM), rest_p0])
    lo = np.concatenate([L([0.05] * NM), rest_lo])
    hi = np.concatenate([L(r2_hi), rest_hi])
if M.has_torso:                               # append fitted R_torso [K/W]
    p0 = np.append(p0, L(1.5)); lo = np.append(lo, L(0.05)); hi = np.append(hi, L(4.0))

r0 = resid(p0)
print("init RMSE=%.2f (n=%d, params=%d, states=%d)%s" %
      (np.sqrt(np.mean(r0 ** 2)), len(r0), len(p0), M.NX,
       (" + steady point '%s' (w=%.0f)" % (SP['tag'], SP['W'])) if SP else ""))
t0 = time.time()
sol = least_squares(resid, p0, bounds=(lo, hi), max_nfev=800)
# report transient and steady RMSE separately
xf = expand(sol.x)                            # full param vector (with fixed R2 if --fix-r2)
M.build(xf)
rt = M.residuals(xf, runs)
print("fit %.0fs  train(transient) RMSE=%.2f C" % (time.time() - t0, np.sqrt(np.mean(rt ** 2))))
if SP is not None:
    rs = M.steady_residual(SP['iq'], SP['R20'], SP['Tamb'], SP['meas'])
    print("            steady-point RMSE=%.2f C  (errors: %s)" %
          (np.sqrt(np.mean(rs ** 2)),
           ' '.join('%s%+.0f' % (x, e) for x, e in
                    zip([x for x in M.ACTS if np.isfinite(SP['meas'][M.ACTS.index(x)])], rs))))

R2, Rl, b = M.unpack(xf)
out = dict(R2=dict(zip(M.ACTS, R2.round(3))),
           R_link=dict(zip(M.ACTS, Rl.round(3))),
           b=dict(zip(M.BSTRUCTS, b.round(2))),
           p=list(xf))
if M.has_torso:
    out['R_torso'] = round(float(M.R_TORSO), 3)
C.save_params(a.limb, out)
print("R2:", out['R2']); print("R_link:", out['R_link']); print("b:", out['b'])
if M.has_torso: print("R_torso:", out['R_torso'], "K/W  (torso @ %.0f C)" % M.torso_temp)
