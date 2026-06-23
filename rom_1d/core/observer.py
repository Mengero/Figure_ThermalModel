"""ThermalObserver — model-based thermistor fault detection + virtual sensor.

Wraps a *built* ThermalROM (does NOT change the fit). Causal, one-step-at-a-time, so the same
logic can run online. Per step:
  1. one-step predict   x_pred = Ad @ x_prev + Bd @ u_prev      (innovation reference)
  2. detect per motor   innovation r_i = Tm_meas_i - x_pred_i ; plus sanity (NaN / out-of-range
                        / flatline-while-active). Hysteresis on the innovation flag.
  3. accommodate        x = x_pred ; clamp HEALTHY motors to their reading (anchor); FAULTED
                        motors keep x_pred  ->  that prediction is the virtual reading.

A clamped stuck sensor masks its own innovation, so the **flatline** check (reading frozen while
the arm is clearly active) is the primary stuck detector; innovation catches dead/NaN/out-of-range
/sudden jumps. A **common-mode** check catches data/logging glitches where every channel jumps
together (physically impossible for independent motor temps) — on such a step no reading is
trusted and the model prediction is held. `forced_drop` marks sensors faulted from boot (used by
the dropout-limit sweep).

innov_thresh is set to 30 °C (not a few °C): under depal cadence the adaptive motor capacitance is
only approximate, so the model can over-predict a healthy motor by ~15-25 °C during sharp lifts.
A tighter band would false-trip a "broken thermistor" warning on a good sensor. 30 °C stays above
that model error while still catching true dead/jumped/out-of-range readings (which the sanity and
flatline checks also catch directly).

Boot: healthy t=0 readings seed their motor node; faulted/missing motors and all structures are
seeded from neighbouring motors (REF = actuators adjacent to a structure in TOPO).
"""
import numpy as np


class ThermalObserver:
    def __init__(self, M, innov_thresh=30.0, persist=3, clear_thresh=4.0, recover=5,
                 t_lo=0.0, t_hi=150.0, flatline_eps=0.5, flatline_min=5.0, active_eps=8.0,
                 cm_jump=5.0, cm_spread=3.0, cm_min_ch=4, forced_drop=()):
        self.M = M
        self.innov_thresh = innov_thresh; self.persist = persist
        self.clear_thresh = clear_thresh; self.recover = recover
        self.t_lo = t_lo; self.t_hi = t_hi
        # stuck/frozen: reading flat (range < flatline_eps) over a ~flatline_min window WHILE the
        # model (open-loop) expects motion > active_eps over that window.
        self.flatline_eps = flatline_eps; self.flatline_min = flatline_min; self.active_eps = active_eps
        # common-mode data glitch: >= cm_min_ch channels jump together by > cm_jump in one step
        # with cross-channel spread < cm_spread (physically impossible -> logging/reference error).
        self.cm_jump = cm_jump; self.cm_spread = cm_spread; self.cm_min_ch = cm_min_ch
        self.forced = set(M.ACTS.index(a) if isinstance(a, str) else a for a in forced_drop)
        # REF: motors adjacent (housing or output) to each structure, for seeding unmeasured nodes
        self.REF = {s: [a for a in M.ACTS if s in M.TOPO[a]] for s in M.STRUCTS}

    def run(self, df, dt):
        M = self.M; NM, NX = M.NM, M.NX
        n = len(df)
        Tamb = df['T_amb'].to_numpy()
        Ttor = (df['T_torso'].to_numpy() if 'T_torso' in df.columns else np.full(n, M.torso_temp)) if M.has_torso else None
        P = np.nan_to_num(df[['P_' + a for a in M.ACTS]].to_numpy())
        QF = M._fet_inputs(P)
        Cser = M.cmotor_series(df)
        Tm = df[['Tm_' + a for a in M.ACTS]].to_numpy(dtype=float)   # measured (may be NaN/garbage)

        def u_of(k):
            parts = [P[k], QF[k], [Tamb[k]]]
            if M.has_torso:
                parts.append([Ttor[k]])
            return np.concatenate(parts)

        # ---- boot state ----
        # only the SURVIVING (healthy, non-dropped) thermistors are known at boot. A dropped
        # sensor contributes nothing -> it does NOT seed itself from its own t=0 reading.
        flags0 = np.array([self._sane(Tm[0, i]) is not None or i in self.forced for i in range(NM)])
        healthy0 = [i for i in range(NM) if not flags0[i] and np.isfinite(Tm[0, i])]
        # no surviving thermistor -> seed everything from the battery/torso temp (the only known
        # body temperature); fall back to ambient if there is no torso boundary.
        gseed = (np.mean([Tm[0, i] for i in healthy0]) if healthy0
                 else (Ttor[0] if Ttor is not None else Tamb[0]))
        reft = {}
        for s in M.STRUCTS:                                  # prefer an adjacent surviving motor,
            vals = [Tm[0, M.ACTS.index(a)] for a in self.REF[s] if M.ACTS.index(a) in healthy0]
            reft[s] = np.mean(vals) if vals else gseed       # else fall back to the survivor mean
        x = np.zeros(NX)
        for i, a in enumerate(M.ACTS):
            x[i] = Tm[0, i] if i in healthy0 else reft[M.TOPO[a][0]]
        for s in M.STRUCTS:
            x[M.SIDX[s]] = reft[s]
        for s in M.FABRICS:
            x[M.FIDX[s]] = reft[s]

        # open-loop shadow (no clamps): used only to judge whether a flat reading is suspicious
        # (flat reading is a fault only if the model expects the node to be MOVING). The motors
        # forced/likely-faulted are driven open-loop here too.
        x_ol = M.simulate(df, dt, x0=x.copy())[:, :NM]
        self._fn = max(2, int(round(self.flatline_min * 60.0 / dt)))   # flatline window in steps

        # ---- outputs ----
        X = np.zeros((n, NX)); X[0] = x
        eff = np.zeros((n, NM)); flags = np.zeros((n, NM), bool)
        innov = np.full((n, NM), np.nan); cause = [['' for _ in range(NM)] for _ in range(n)]
        cm = np.zeros(n, bool)                                        # common-mode data-glitch step
        cnt = np.zeros(NM, int); rec = np.zeros(NM, int); faulted = flags0.copy()
        # step 0 bookkeeping
        for i in range(NM):
            flags[0, i] = faulted[i]
            eff[0, i] = x[i] if faulted[i] else Tm[0, i]
            if faulted[i]:
                cause[0][i] = 'boot'

        for k in range(1, n):
            Ad, Bd = M._op_for_Cm(Cser[k - 1], dt)
            x_pred = Ad @ x + Bd @ u_of(k - 1)
            cm[k] = self._commonmode(Tm, k)        # all channels jump together -> data glitch, trust none
            for i in range(NM):
                m = Tm[k, i]; r = m - x_pred[i] if np.isfinite(m) else np.nan
                innov[k, i] = r
                if cm[k]:                                           # common-mode glitch: predict through,
                    cnt[i] = 0; cause[k][i] = 'commonmode'          # don't latch a per-sensor fault
                    flags[k, i] = faulted[i]
                    continue
                c = self._sane(m)                                   # NaN / OOR (instant)
                if c is None and self._flatline(Tm, x_ol, k, i):
                    c = 'flatline'
                if c is None and np.isfinite(r) and abs(r) > self.innov_thresh:
                    cnt[i] += 1
                    if cnt[i] >= self.persist:
                        c = 'innovation'
                else:
                    cnt[i] = 0
                if i in self.forced:
                    c = 'forced'
                # update fault state with recovery hysteresis
                if c is not None:
                    faulted[i] = True; rec[i] = 0
                elif faulted[i] and i not in self.forced:
                    rec[i] = rec[i] + 1 if (np.isfinite(r) and abs(r) < self.clear_thresh) else 0
                    if rec[i] >= self.recover:
                        faulted[i] = False
                flags[k, i] = faulted[i]; cause[k][i] = c or ''
            # accommodate: predict, then clamp the healthy motors to their reading. On a common-mode
            # glitch step trust NO reading (all corrupt) -> hold the model prediction for every motor.
            x = x_pred.copy()
            for i in range(NM):
                use_meas = (not cm[k]) and (not faulted[i]) and np.isfinite(Tm[k, i])
                if use_meas:
                    x[i] = Tm[k, i]
                eff[k, i] = Tm[k, i] if use_meas else x_pred[i]
            X[k] = x
        return dict(t=df['t_s'].to_numpy() / 60.0, X=X, eff=eff, flags=flags,
                    innov=innov, meas=Tm, cause=cause, cm=cm)

    # ---------- detectors ----------
    def _sane(self, m):
        if not np.isfinite(m):
            return 'nan'
        if m < self.t_lo or m > self.t_hi:
            return 'oor'
        return None

    def _commonmode(self, Tm, k):
        """one-step jump shared by most channels -> physically impossible, a logging/reference glitch.
        Real motor temps move independently with their own power; a step where >= cm_min_ch channels
        all jump by > cm_jump with cross-channel spread < cm_spread is a data error, not a thermal one."""
        d = Tm[k] - Tm[k - 1]
        d = d[np.isfinite(d)]
        if len(d) < self.cm_min_ch:
            return False
        big = np.abs(d) > self.cm_jump
        if big.sum() < self.cm_min_ch:
            return False
        db = d[big]
        return (np.max(db) - np.min(db)) < self.cm_spread     # all jumped together, same sign+size

    def _flatline(self, Tm, x_ol, k, i):
        """reading frozen over the window while the MODEL (open-loop) expects the node to move.
        Gating on the model's expected motion (not other sensors) avoids false alarms on a
        genuinely idle/plateaued joint — there x_ol is flat too, so no flag.

        active_eps is 8 C (not ~2): the open-loop shadow over-predicts by up to ~5.4 C over a
        5 min window in depal/UPS, so a smaller gate would call a real plateaued reading "stuck".
        8 C clears that model error while still catching a sensor frozen during genuine heating
        (where the model expects far more than 8 C of motion)."""
        if k < self._fn:
            return False
        w = Tm[k - self._fn:k + 1, i]
        if not np.all(np.isfinite(w)) or (np.max(w) - np.min(w)) >= self.flatline_eps:
            return False
        wo = x_ol[k - self._fn:k + 1, i]
        return (np.max(wo) - np.min(wo)) > self.active_eps
