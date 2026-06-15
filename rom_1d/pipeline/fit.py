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
p0 = np.concatenate([L([0.7] * NM), L([5] * NM), L([10.0] * M.NB)])
lo = np.concatenate([L([0.05] * NM), L([0.05] * NM), L([1.0] * M.NB)])
hi = np.concatenate([L([1.35] * NM), L([15] * NM), L([40] * M.NB)])

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
C.save_params(a.limb, out)
print("R2:", out['R2']); print("R_link:", out['R_link']); print("b:", out['b'])
