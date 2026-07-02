"""ThermalROM — the limb-agnostic 1D lumped-parameter thermal engine.

Given a LimbConfig it builds an N-state RC network (NM motor + NS structure +
NF fabric nodes), discretizes with a zero-order hold (Ad=expm(A*dt),
Bd=A^-1(Ad-I)B), and rolls it forward.  Motor capacitance is time-varying: a
sudden |dP/dt| transient dips C to the winding mass, and it recovers toward the
full assembly mass with time constant tau (steady or slowly-ramping power -> full C).

Grey-box fit parameters (log10-encoded): R2 (per actuator), R_link (per actuator),
b (per externally-exposed surface) = 2*NM + NB total.  All capacitances are fixed.

Nothing here is arm- or leg-specific; everything specific lives in the LimbConfig.
"""
import numpy as np
from scipy.linalg import expm


class ThermalROM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ACTS, self.STRUCTS, self.FABRICS = cfg.ACTS, cfg.STRUCTS, cfg.FABRICS
        self.TOPO, self.TS_COLS = cfg.TOPO, cfg.TS_COLS
        self.NM, self.NS, self.NF = len(cfg.ACTS), len(cfg.STRUCTS), len(cfg.FABRICS)
        self.NX = self.NM + self.NS + self.NF
        self.SIDX = {s: self.NM + k for k, s in enumerate(cfg.STRUCTS)}
        self.FIDX = {s: self.NM + self.NS + k for k, s in enumerate(cfg.FABRICS)}

        # observed node per measured structure: fabric state if covered, else metal
        self.OBS_NODE = {}
        for s in cfg.TS_COLS:
            self.OBS_NODE[s] = self.FIDX[s] if s in cfg.FABRICS else self.SIDX[s]

        # bare structures = exposed, not fabric-covered, not enclosed
        self.BARE = [s for s in cfg.STRUCTS if s not in cfg.FABRICS and s not in cfg.ENCLOSED]
        # externally-exposed surfaces (each gets its own convective b): fabrics then bare
        self.BSTRUCTS = list(cfg.FABRICS) + self.BARE
        self.NB = len(self.BSTRUCTS)

        self.boundary_idx = [cfg.ACTS.index(a) for a in cfg.BOUNDARY_ACTS]
        self.free_mot = [i for i in range(self.NM) if i not in self.boundary_idx]

        self.C_M, self.C_WIND = cfg.C_M, cfg.C_WIND
        self.C_FIX = np.array([cfg.C_S_FIX[s] for s in cfg.STRUCTS] +
                              [cfg.C_F_FIX[s] for s in cfg.FABRICS])
        self.QFET = cfg.QFET
        # extra constant loads -> per-actuator vector added to the FET inputs
        self.extra_q = np.zeros(self.NM)
        for a, q in cfg.EXTRA_Q.items():
            self.extra_q[cfg.ACTS.index(a)] = q

        # optional torso boundary: a fixed-temperature sink on one structure,
        # coupled through a fitted R_torso (an extra parameter appended to p).
        self.has_torso = cfg.TORSO is not None
        if self.has_torso:
            tn = cfg.TORSO[0]                       # may be an actuator (motor node) or a structure
            self.torso_node = self.ACTS.index(tn) if tn in self.ACTS else self.SIDX[tn]
            self.torso_temp = float(cfg.TORSO[1])
        self.fit_rstack = getattr(cfg, 'FIT_RSTACK', False)
        self.n_rstack = self.NF if self.fit_rstack else 0
        self.n_in = 2 * self.NM + 1 + (1 if self.has_torso else 0)   # width of u
        self.NP = 2 * self.NM + self.NB + self.n_rstack + (1 if self.has_torso else 0)   # number of fit params

        self._opcache = {}
        self.G_MAT = self.BQ_MAT = self.R2_VEC = self.B_COEF = self.R_TORSO = None

    # ---------- parameters ----------
    def unpack(self, p):
        """log10 params -> (R2[NM], R_link[NM], b[NB])."""
        NM = self.NM
        return 10 ** p[0:NM], 10 ** p[NM:2 * NM], 10 ** p[2 * NM:2 * NM + self.NB]

    # ---------- adaptive motor capacitance ----------
    def cmotor_series(self, df):
        """per-step motor C [J/K], shape (n, NM), from this run's |dP/dt| transients.

        dip g in [0,1] (0 = full assembly C, 1 = winding C).  Power before the run
        is assumed 0, so the start is a power-on transient.
            event[k] = clip(|dP/dt| / DPDT_THRESH, 0, 1)
            g[k]     = max(g[k-1]*exp(-dt/tau), event[k])
            f = 1 - g ;  C = C_wind + (C_full - C_wind)*f
        """
        cfg = self.cfg
        P = np.nan_to_num(df[['P_' + a for a in self.ACTS]].to_numpy())
        t = df['t_s'].to_numpy(); n = len(t)
        dtv = np.empty(n); dtv[1:] = np.diff(t); dtv[0] = dtv[1] if n > 1 else 1.0
        dP = np.abs(np.diff(P, axis=0, prepend=np.zeros((1, self.NM))))   # dP[0]=P[0]
        ev = np.clip((dP / dtv[:, None]) / cfg.DPDT_THRESH, 0.0, 1.0)
        g = np.zeros((n, self.NM)); g[0] = ev[0]
        decay = np.exp(-dtv / cfg.CADAPT_TAU)
        for k in range(1, n):
            g[k] = np.maximum(g[k - 1] * decay[k], ev[k])
        f = (1.0 - g) if cfg.ADAPT_C else np.ones((n, self.NM))
        return self.C_WIND + (self.C_M - self.C_WIND) * f

    # ---------- network assembly ----------
    def _mats_from_C(self, Cm):
        """continuous A,B for a motor-C vector Cm (structure/fabric C fixed)."""
        Cinv = 1.0 / np.concatenate([Cm, self.C_FIX])
        A = self.G_MAT * Cinv[:, None]
        B = self.BQ_MAT * Cinv[:, None]
        B[:self.NM, :self.NM] = np.diag(Cinv[:self.NM])
        return A, B

    def build(self, p):
        """assemble the conductance network for params p; returns full-C (Ad-ready) A,B."""
        cfg = self.cfg; NM, NX = self.NM, self.NX
        R2, Rl, bvec = self.unpack(p)
        self.B_COEF = dict(zip(self.BSTRUCTS, bvec)); self.R2_VEC = R2
        G = np.zeros((NX, NX)); Bq = np.zeros((NX, self.n_in))   # u=[P(NM),QF(NM),Tamb(,Ttorso)]

        def couple(n1, n2, R):
            G[n1, n1] -= 1 / R
            if n2 == 'amb':
                Bq[n1, 2 * NM] += 1 / R
            else:
                G[n1, n2] += 1 / R; G[n2, n1] += 1 / R; G[n2, n2] -= 1 / R

        for i, a in enumerate(self.ACTS):
            hs, os_ = self.TOPO[a]; o = self.SIDX[os_]; h = self.SIDX[hs]
            couple(i, h, R2[i])
            if o != h:
                couple(h, o, Rl[i])      # housing==output (boundary act) -> no link
            Bq[h, NM + i] += 1.0         # FET (+extra load) into housing structure
        self.RSTACK = ({s: 10 ** p[2 * NM + self.NB + k] for k, s in enumerate(self.FABRICS)}
                       if self.fit_rstack else dict(cfg.R_STACK))
        for s in self.FABRICS:
            couple(self.SIDX[s], self.FIDX[s], self.RSTACK[s])
            couple(self.FIDX[s], 'amb', 1 / (self.B_COEF[s] * cfg.AREA[s]))
        for s in self.BARE:
            couple(self.SIDX[s], 'amb', 1 / (self.B_COEF[s] * cfg.AREA[s]))
        # enclosed structures: no ambient, no extra link (anchored only via their actuators)
        if self.has_torso:
            self.R_TORSO = 10 ** p[2 * NM + self.NB + self.n_rstack]          # fitted torso-link resistance [K/W]
            G[self.torso_node, self.torso_node] -= 1 / self.R_TORSO
            Bq[self.torso_node, 2 * NM + 1] += 1 / self.R_TORSO   # fixed torso temp = column 2NM+1
        self.G_MAT, self.BQ_MAT = G, Bq; self._opcache.clear()
        return self._mats_from_C(self.C_M)

    def _op_for_Cm(self, Cm, dt):
        """discrete (Ad,Bd) for motor-C vector Cm, cached by (dt, quantized motor C)."""
        key = (round(dt, 6), tuple(np.round(Cm, 1)))
        op = self._opcache.get(key)
        if op is None:
            A, B = self._mats_from_C(Cm)
            Ad = expm(A * dt); Bd = np.linalg.solve(A, (Ad - np.eye(self.NX))) @ B
            op = (Ad, Bd); self._opcache[key] = op
        return op

    # ---------- forward simulation ----------
    def _fet_inputs(self, P):
        """FET (always-on) + per-actuator extra constant load, shape like P."""
        return np.full_like(P, self.QFET) + self.extra_q[None, :]

    def simulate(self, df, dt, x0=None):
        cfg = self.cfg; NM, NX = self.NM, self.NX
        Tamb_t = df['T_amb'].to_numpy()
        Cser = self.cmotor_series(df)
        P = np.nan_to_num(df[['P_' + a for a in self.ACTS]].to_numpy())
        QF = self._fet_inputs(P)
        n = len(df); X = np.zeros((n, NX)); x = np.zeros(NX)
        # torso boundary: per-step from a 'T_torso' column if present, else the fixed cfg value
        Ttor_t = ((df['T_torso'].to_numpy() if 'T_torso' in df.columns else np.full(n, self.torso_temp))
                  if self.has_torso else None)

        def _u(k):
            parts = [P[k], QF[k], [Tamb_t[k]]]
            if self.has_torso:
                parts.append([Ttor_t[k]])
            return np.concatenate(parts)

        if x0 is not None:                                # caller-supplied initial state
            x = np.asarray(x0, float).copy()
            Tb = {i: df['Tm_' + self.ACTS[i]].to_numpy() for i in self.boundary_idx}
            X[0] = x
            for k in range(n - 1):
                Ad, Bd = self._op_for_Cm(Cser[k], dt)
                x = Ad @ x + Bd @ _u(k)
                for i, arr in Tb.items():
                    if np.isfinite(arr[k + 1]):
                        x[i] = arr[k + 1]
                X[k + 1] = x
            return X

        # init structure metal from the measured fabric/bare TC via the steady flux balance:
        #   T_struct = T_TC + (T_TC - T_amb)*b*A*R_stack   (bare: R_stack=0 -> T_struct=T_TC)
        Ts0 = {s: (df[self.TS_COLS[s]].iloc[0] if s in self.TS_COLS else np.nan) for s in self.STRUCTS}
        metal0 = {}
        for s in self.STRUCTS:
            if np.isfinite(Ts0[s]):
                Q0 = max(Ts0[s] - Tamb_t[0], 0.0) * (self.B_COEF[s] * cfg.AREA[s])
                metal0[s] = Ts0[s] + Q0 * self.RSTACK.get(s, 0.0)
            else:
                metal0[s] = None
        for s, src in cfg.ENCLOSED.items():
            metal0[s] = metal0[src]                       # enclosed inherits neighbour metal
        for s in self.STRUCTS:
            if metal0[s] is None:                         # no TC and not enclosed: start at ambient
                metal0[s] = Tamb_t[0]
            x[self.SIDX[s]] = metal0[s]
        for s in self.FABRICS:
            x[self.FIDX[s]] = Ts0[s]
        for i, a in enumerate(self.ACTS):
            tm = df['Tm_' + a].iloc[0]; hs, _ = self.TOPO[a]
            x[i] = tm if np.isfinite(tm) else x[self.SIDX[hs]]

        # boundary actuators: clamp to measured Tm every step
        Tb = {i: df['Tm_' + self.ACTS[i]].to_numpy() for i in self.boundary_idx}
        for i, arr in Tb.items():
            if np.isfinite(arr[0]):
                x[i] = arr[0]
        X[0] = x
        for k in range(n - 1):
            Ad, Bd = self._op_for_Cm(Cser[k], dt)
            x = Ad @ x + Bd @ _u(k)
            for i, arr in Tb.items():
                if np.isfinite(arr[k + 1]):
                    x[i] = arr[k + 1]
            X[k + 1] = x
        return X

    def struct_obs(self, X, df):
        """predicted TC readings: fabric state where covered, metal state where bare."""
        return {s: X[:, self.OBS_NODE[s]] for s in self.TS_COLS}

    # ---------- energy ----------
    def energy_flows(self, X, df):
        """instantaneous heat-flow rates [W] vs time (call after build()).

        Returns dict of arrays + t in minutes.  Boundary actuators are treated as
        sources feeding their housing; their stored energy and power generation
        are excluded from the in-model totals.  Motor storage uses time-varying C.
        """
        cfg = self.cfg
        t = df['t_s'].to_numpy(); Tamb = df['T_amb'].to_numpy()
        P = np.nan_to_num(df[['P_' + a for a in self.ACTS]].to_numpy())
        QF = self._fet_inputs(P)
        Qgen = P[:, self.free_mot].sum(1) + QF.sum(1)         # boundary power excluded
        Qbnd = np.zeros(len(t))
        for i in self.boundary_idx:
            hs, _ = self.TOPO[self.ACTS[i]]
            Qbnd += (df['Tm_' + self.ACTS[i]].to_numpy() - X[:, self.SIDX[hs]]) / self.R2_VEC[i]
        Qamb = np.zeros(len(t))
        for s in self.FABRICS:
            Qamb += (X[:, self.FIDX[s]] - Tamb) * (self.B_COEF[s] * cfg.AREA[s])
        for s in self.BARE:
            Qamb += (X[:, self.SIDX[s]] - Tamb) * (self.B_COEF[s] * cfg.AREA[s])
        dT = np.gradient(X, t, axis=0)
        Cmt = self.cmotor_series(df)
        Qm = (dT[:, self.free_mot] * Cmt[:, self.free_mot]).sum(1)
        Qs = (sum(cfg.C_S_FIX[s] * dT[:, self.SIDX[s]] for s in self.STRUCTS) +
              sum(cfg.C_F_FIX[s] * dT[:, self.FIDX[s]] for s in self.FABRICS))
        Qtorso = ((X[:, self.torso_node] - self.torso_temp) / self.R_TORSO
                  if self.has_torso else np.zeros(len(t)))
        return dict(t=t / 60.0, gen=Qgen, bnd=Qbnd, mot=Qm, struct=Qs, amb=Qamb, torso=Qtorso)

    # ---------- steady state ----------
    def steady_state(self, P, Tamb):
        """solve all node temps at steady state given motor copper power P [W] (len NM)
        and ambient [C]; all motors driven (no clamping). Capacitance irrelevant here.
        Build() must have been called. Torso (if any) is held at its fixed temp."""
        NM = self.NM
        QF = self._fet_inputs(np.asarray(P)[None, :])[0]
        rhs = self.BQ_MAT[:, NM:2 * NM] @ QF + self.BQ_MAT[:, 2 * NM] * Tamb
        if self.has_torso:
            rhs = rhs + self.BQ_MAT[:, 2 * NM + 1] * self.torso_temp
        rhs[:NM] = rhs[:NM] + np.asarray(P)
        return np.linalg.solve(self.G_MAT, -rhs)

    def steady_residual(self, iq, R20, Tamb, meas):
        """steady-state operating-point residual: predicted − measured motor temp [C],
        for instrumented motors. Build() must have been called. Self-consistent copper
        power (P depends on winding temp). meas/iq/R20 are per-ACTS arrays (meas may be NaN)."""
        iq = np.asarray(iq); R20 = np.asarray(R20); meas = np.asarray(meas)
        Tw = np.where(np.isfinite(meas), meas, 80.0)
        for _ in range(60):
            P = 1.5 * iq ** 2 * R20 * (234.5 + Tw) / (234.5 + 20.0)
            T = self.steady_state(P, Tamb)
            if np.max(np.abs(T[:self.NM] - Tw)) < 1e-4:
                Tw = T[:self.NM]; break
            Tw = T[:self.NM]
        ok = np.isfinite(meas)
        return T[:self.NM][ok] - meas[ok]

    # ---------- fitting ----------
    def residuals(self, p, runs, dt=2.0):
        self.build(p)
        res = []
        for df in runs:
            X = self.simulate(df, dt)
            for i, a in enumerate(self.ACTS):
                if i in self.boundary_idx:
                    continue                              # measured boundary, not a target
                m = df['Tm_' + a].to_numpy(); ok = np.isfinite(m)
                res.append(X[ok, i] - m[ok])
            obs = self.struct_obs(X, df)
            for s, c in self.TS_COLS.items():
                res.append(obs[s] - df[c].to_numpy())
        return np.concatenate(res)
