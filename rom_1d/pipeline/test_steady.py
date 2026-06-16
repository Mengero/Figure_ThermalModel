"""Held-out steady-state validation: drive every motor at its measured continuous
current and predict the steady-state winding temps, vs the measured values.

    python test_steady.py --limb arm

This is a TEST point (not used in fitting). Copper power depends on winding temp,
  P_i = 1.5 * iq_i^2 * R20_i * (234.5 + Tw_i)/(234.5 + 20),
so we solve self-consistently: guess Tw -> P -> steady-state solve -> update Tw, iterate.
Uses the fitted network (incl. the torso sink if the limb has one).
"""
import argparse, json, os
import numpy as np
import _common as C

ap = argparse.ArgumentParser()
ap.add_argument('--limb', default='arm')
ap.add_argument('--case', default='steady_061526')
a = ap.parse_args()

M = C.rom(a.limb)
M.build(C.load_params(a.limb))
d = json.load(open(os.path.join(C.DATA_DIR, a.limb, a.case + '.json')))
Tamb = d['T_amb']
iq = np.array([d['iq'][x] for x in M.ACTS])
R20 = np.array([M.cfg.R20[x] for x in M.ACTS])
meas = {k: v for k, v in d['Tm'].items()}

# self-consistent steady state (copper R depends on winding temp)
Tw = np.array([meas.get(x, 80.0) for x in M.ACTS])      # warm start
for _ in range(50):
    P = 1.5 * iq ** 2 * R20 * (234.5 + Tw) / (234.5 + 20.0)
    T = M.steady_state(P, Tamb)
    if np.max(np.abs(T[:M.NM] - Tw)) < 1e-4:
        Tw = T[:M.NM]; break
    Tw = T[:M.NM]

torso = (' torso=%.0fC R_torso=%.2f' % (M.torso_temp, M.R_TORSO)) if M.has_torso else ' (no torso)'
print('Steady-state TEST: %s  (ambient %.1f C,%s)\n' % (a.case, Tamb, torso))
print('%-7s %7s %8s %8s %7s' % ('joint', 'iq[A]', 'model', 'meas', 'err'))
errs = []
for i, act in enumerate(M.ACTS):
    m = meas.get(act, np.nan)
    e = T[i] - m
    if np.isfinite(m):
        errs.append(e)
    print('%-7s %7.1f %8.1f %8.1f %7.1f' % (act, iq[i], T[i], m, e))
errs = np.array(errs)
print('\nmotor RMSE = %.2f C   mean |err| = %.2f C' % (np.sqrt(np.mean(errs ** 2)), np.mean(np.abs(errs))))
print('\nhand structure temp = %.1f C' % T[M.SIDX['hand']])
print('structure temps [C]:', {s: round(T[M.SIDX[s]], 1) for s in M.STRUCTS})
