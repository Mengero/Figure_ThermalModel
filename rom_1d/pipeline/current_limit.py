"""Continuous current limit per actuator (steady state) + transient climb to it.

    python current_limit.py --limb arm --tamb 25

Scenario: ALL actuators driven simultaneously, every motor winding held at T_LIMIT
(120 C) at steady state — the coupled whole-limb worst case. Electronics heat
(FET per joint + any extra load) included. Capacitance is irrelevant at steady
state, so the limit itself uses only the fitted R network (R2, R_link, b).

Steady-state method:
  * fix all motor temps = T_LIMIT, solve the structure+fabric network for their temps
  * each motor's required copper loss  P_i = (T_LIMIT - T_housing_i) / R2_i
  * back out the continuous q-axis current:
        P = 1.5 * iq^2 * R20 * (234.5 + T_LIMIT)/(234.5 + 20)
     => iq = sqrt( P / (1.5 * R20 * kT) )

Transient: drive every motor with that constant continuous current from a cold
(ambient) start and integrate forward. Copper power rises with winding temperature,
  P_i(t) = P_cu_i * (234.5 + Tw_i)/(234.5 + T_LIMIT),
so each motor converges exactly to its T_LIMIT steady state — the plot shows how
fast each one gets there. Full assembly heat mass is used (slow ramp, no dip).

Result is governed by headroom dT = T_LIMIT - T_amb; change --tamb to rescale.
Outputs: figures/<limb>_current_limit_transient.png
"""
import argparse
import numpy as np
from scipy.linalg import expm
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import _common as C

ap = argparse.ArgumentParser()
ap.add_argument('--limb', default='arm')
ap.add_argument('--tlimit', type=float, default=120.0)
ap.add_argument('--tamb', type=float, default=25.0)
ap.add_argument('--dt', type=float, default=5.0, help='transient step [s]')
a = ap.parse_args()

M = C.rom(a.limb)
if not M.cfg.R20:
    raise SystemExit(f"limb '{a.limb}' has no R20 map — fill cfg.R20 before running current_limit.")
M.build(C.load_params(a.limb))
G, Bq, R2 = M.G_MAT, M.BQ_MAT, M.R2_VEC
mot = np.arange(M.NM); free = np.arange(M.NM, M.NX)
QF = M._fet_inputs(np.zeros((1, M.NM)))[0]                 # FET + extra loads per actuator

# ---- steady state: all motors fixed at T_LIMIT, solve the rest ----
u_ss = np.concatenate([np.zeros(M.NM), QF, [a.tamb]])
Tm = np.full(M.NM, a.tlimit)
rhs = G[np.ix_(free, mot)] @ Tm + Bq[free, :] @ u_ss
Tfree = np.linalg.solve(G[np.ix_(free, free)], -rhs)
T = np.zeros(M.NX); T[mot] = a.tlimit; T[free] = Tfree

kT = (234.5 + a.tlimit) / (234.5 + 20.0)
Pcu = np.zeros(M.NM); iq = np.zeros(M.NM)
print(f"\nContinuous current limit  (all motors @ {a.tlimit:.0f} C, ambient {a.tamb:.0f} C, "
      f"dT={a.tlimit-a.tamb:.0f} C)\n")
print(f"{'joint':8s} {'motor':14s} {'housing':9s} {'T_house':>8s} {'P_cu':>8s} {'R(T)':>7s} {'iq':>7s}")
print(f"{'':8s} {'':14s} {'':9s} {'[C]':>8s} {'[W]':>8s} {'[ohm]':>7s} {'[A]':>7s}")
for i, act in enumerate(M.ACTS):
    hs, _ = M.TOPO[act]; Th = T[M.SIDX[hs]]
    r20 = M.cfg.R20[act]
    Pcu[i] = max((a.tlimit - Th) / R2[i], 0.0)
    iq[i] = np.sqrt(Pcu[i] / (1.5 * r20 * kT))
    print(f"{act:8s} {M.cfg.MOTOR.get(act,''):14s} {hs:9s} {Th:8.1f} {Pcu[i]:8.1f} "
          f"{r20*kT:7.3f} {iq[i]:7.1f}")
print()
print("structure temps [C]:", {s: round(T[M.SIDX[s]], 1) for s in M.STRUCTS})

# ---- transient: constant continuous current from ambient -> steady state ----
A, B = M._mats_from_C(M.C_M)                              # full assembly heat mass
evr = np.linalg.eigvals(A).real
tau_slow = -1.0 / evr[evr < 0].max()                     # slowest mode (closest to 0)
t_max = min(6.0 * tau_slow, 8 * 3600.0)                  # cover the slow mode (~6 tau), cap at 8 h
n = int(t_max / a.dt) + 1
Ad = expm(A * a.dt); Bd = np.linalg.solve(A, (Ad - np.eye(M.NX))) @ B
x = np.full(M.NX, a.tamb); X = np.zeros((n, M.NX))
for k in range(n):
    X[k] = x
    P = Pcu * (234.5 + x[mot]) / (234.5 + a.tlimit)      # constant iq -> P(Tw)
    x = Ad @ x + Bd @ np.concatenate([P, QF, [a.tamb]])
t = np.arange(n) * a.dt / 60.0                           # minutes

# time for each motor to reach 95% of its rise toward the limit
rise = a.tlimit - a.tamb
t95 = {}
for i, act in enumerate(M.ACTS):
    if Pcu[i] <= 0:
        continue
    hit = np.where(X[:, i] - a.tamb >= 0.95 * rise)[0]
    t95[act] = t[hit[0]] if len(hit) else np.nan

# ---- plot ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 11), sharex=True)
palette = plt.cm.tab10.colors
for i, act in enumerate(M.ACTS):
    if Pcu[i] <= 0:
        continue
    lab = f"{act} (iq={iq[i]:.1f}A" + (f", t95={t95[act]:.0f}min)" if np.isfinite(t95.get(act, np.nan)) else ")")
    ax1.plot(t, X[:, i], lw=1.8, color=palette[i % 10], label=lab)
ax1.axhline(a.tlimit, color='k', ls=':', lw=1.3, label=f'{a.tlimit:.0f} C limit')
ax1.set_ylabel('motor winding T [°C]')
ax1.set_title(f'{a.limb} — climb to continuous-current steady state '
              f'(all motors driven, ambient {a.tamb:.0f} C)')
ax1.grid(alpha=.3); ax1.legend(fontsize=8, ncol=2, loc='lower right')

for j, s in enumerate(M.STRUCTS):
    ax2.plot(t, X[:, M.SIDX[s]], lw=1.6, color=palette[j % 10], label=s)
    ax2.axhline(T[M.SIDX[s]], color=palette[j % 10], ls='--', lw=.8, alpha=.5)
ax2.set_ylabel('structure T [°C]'); ax2.set_xlabel('time [min]')
ax2.set_title('structure nodes (dashed = analytic steady state)')
ax2.grid(alpha=.3); ax2.legend(fontsize=8, ncol=2, loc='lower right')

out = C.fig_path(f'{a.limb}_current_limit_transient.png')
plt.tight_layout(); plt.savefig(out, dpi=600)
print(f"\nslowest thermal time constant ~ {tau_slow/60:.0f} min; integrated {t[-1]:.0f} min")
print(f"final motor T range: {X[-1, mot].min():.1f}–{X[-1, mot].max():.1f} C (target {a.tlimit:.0f})")
print(f"saved {out}")
