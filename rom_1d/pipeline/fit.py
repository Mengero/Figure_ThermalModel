"""Grey-box fit of R2, R_link, b for one limb.

    python fit.py --limb arm

Writes fits/<limb>_fitted_params.json. Capacitances are fixed (not fit); the
adaptive motor C is handled inside the engine.
"""
import argparse, time
import numpy as np
from scipy.optimize import least_squares
import _common as C

ap = argparse.ArgumentParser()
ap.add_argument('--limb', default='arm')
a = ap.parse_args()

M = C.rom(a.limb)
runs = [C.load(a.limb, t) for t in M.cfg.TRAIN]
L = np.log10
NM = M.NM
p0 = np.concatenate([L([0.7] * NM), L([5] * NM), L([20] * M.NB)])
lo = np.concatenate([L([0.05] * NM), L([0.0001] * NM), L([6] * M.NB)])
r2_hi = [1.4] * NM
r2_hi[M.ACTS.index('J1')] = 1.0               # J1 R2 capped tighter; others 1.4
hi = np.concatenate([L(r2_hi), L([15] * NM), L([40] * M.NB)])
if M.has_torso:                               # append fitted R_torso [K/W]
    p0 = np.append(p0, L(1.5)); lo = np.append(lo, L(0.05)); hi = np.append(hi, L(4.5))

r0 = M.residuals(p0, runs)
print("init RMSE=%.2f (n=%d, params=%d, states=%d)" %
      (np.sqrt(np.mean(r0 ** 2)), len(r0), len(p0), M.NX))
t0 = time.time()
sol = least_squares(M.residuals, p0, args=(runs,), bounds=(lo, hi), max_nfev=800)
print("fit %.0fs  train RMSE=%.2f C" % (time.time() - t0, np.sqrt(np.mean(sol.fun ** 2))))

R2, Rl, b = M.unpack(sol.x)
out = dict(R2=dict(zip(M.ACTS, R2.round(3))),
           R_link=dict(zip(M.ACTS, Rl.round(3))),
           b=dict(zip(M.BSTRUCTS, b.round(2))),
           p=list(sol.x))
if M.has_torso:
    out['R_torso'] = round(float(M.R_TORSO), 3)
C.save_params(a.limb, out)
print("R2:", out['R2']); print("R_link:", out['R_link']); print("b:", out['b'])
if M.has_torso: print("R_torso:", out['R_torso'], "K/W  (torso @ %.0f C)" % M.torso_temp)
