"""Continuous current limit per actuator (steady-state thermal back-calculation).

    python current_limit.py --limb arm --tamb 25

Scenario: ALL actuators driven simultaneously, every motor winding held at T_LIMIT
(120 C) at steady state — the coupled whole-limb worst case. Electronics heat
(FET per joint + any extra load) included. Capacitance is irrelevant at steady
state, so this uses only the fitted R network (R2, R_link, b).

Method:
  * fix all motor temps = T_LIMIT, solve the structure+fabric network for their temps
  * each motor's required copper loss  P_i = (T_LIMIT - T_housing_i) / R2_i
  * back out the continuous q-axis current:
        P = 1.5 * iq^2 * R20 * (234.5 + T_LIMIT)/(234.5 + 20)
     => iq = sqrt( P / (1.5 * R20 * kT) )

Result is governed by headroom dT = T_LIMIT - T_amb; change --tamb to rescale.
"""
import argparse
import numpy as np
import _common as C

ap = argparse.ArgumentParser()
ap.add_argument('--limb', default='arm')
ap.add_argument('--tlimit', type=float, default=120.0)
ap.add_argument('--tamb', type=float, default=25.0)
a = ap.parse_args()

M = C.rom(a.limb)
if not M.cfg.R20:
    raise SystemExit(f"limb '{a.limb}' has no R20 map — fill cfg.R20 before running current_limit.")
M.build(C.load_params(a.limb))
G, Bq, R2 = M.G_MAT, M.BQ_MAT, M.R2_VEC
mot = np.arange(M.NM); free = np.arange(M.NM, M.NX)

QF = M._fet_inputs(np.zeros((1, M.NM)))[0]                 # FET + extra loads per actuator
u = np.concatenate([np.zeros(M.NM), QF, [a.tamb]])
Tm = np.full(M.NM, a.tlimit)
rhs = G[np.ix_(free, mot)] @ Tm + Bq[free, :] @ u
Tfree = np.linalg.solve(G[np.ix_(free, free)], -rhs)
T = np.zeros(M.NX); T[mot] = a.tlimit; T[free] = Tfree

kT = (234.5 + a.tlimit) / (234.5 + 20.0)
print(f"\nContinuous current limit  (all motors @ {a.tlimit:.0f} C, ambient {a.tamb:.0f} C, "
      f"dT={a.tlimit-a.tamb:.0f} C)\n")
print(f"{'joint':8s} {'motor':14s} {'housing':9s} {'T_house':>8s} {'P_cu':>8s} {'R(T)':>7s} {'iq':>7s}")
print(f"{'':8s} {'':14s} {'':9s} {'[C]':>8s} {'[W]':>8s} {'[ohm]':>7s} {'[A]':>7s}")
for i, act in enumerate(M.ACTS):
    hs, _ = M.TOPO[act]; Th = T[M.SIDX[hs]]
    r20 = M.cfg.R20[act]
    Pcu = (a.tlimit - Th) / R2[i]
    iq = np.sqrt(max(Pcu, 0.0) / (1.5 * r20 * kT))
    print(f"{act:8s} {M.cfg.MOTOR.get(act,''):14s} {hs:9s} {Th:8.1f} {Pcu:8.1f} "
          f"{r20*kT:7.3f} {iq:7.1f}")
print()
print("structure temps [C]:", {s: round(T[M.SIDX[s]], 1) for s in M.STRUCTS})
