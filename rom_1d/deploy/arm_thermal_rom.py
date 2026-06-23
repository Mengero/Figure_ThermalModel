"""
Figure F03 LEFT-ARM thermal ROM — single-file, fitted, ready to embed.
================================================================================

A 1-D lumped-parameter (RC-network) thermal model of the left arm. Given each
actuator's q-axis current (or copper power), the ambient temperature, and the
battery/torso temperature, it predicts all seven MOTOR winding temperatures (and
the intermediate structure/fabric temperatures) by integrating an 18-state RC
network forward in time.

It is FITTED: the parameters baked in below were identified from logged robot
data (UPS + depalletizing runs). Open-loop motor RMSE ~4-5 C.

Two ways to use it
------------------
  ONLINE  (robot tick loop):  rom = ArmThermalROM();  x = rom.init_state(...)
                              x = rom.step(x, iq, Tamb, Tbatt, dt[, clamp=...])
  OFFLINE (a logged run):     X = rom.simulate(iq_series, Tamb, Tbatt, dt, x0)

`clamp` lets you anchor any motor whose thermistor is healthy to its measured
temperature; un-clamped motors are pure model predictions (this is exactly the
virtual-sensor / sensor-dropout behaviour).

Units: temperature [degC], power [W], current [A], resistance [K/W] (thermal)
       or [ohm] (winding R20), capacitance [J/K], time [s], area [m^2].

Dependencies: numpy. (Uses scipy.linalg.expm if present; falls back to a numpy
matrix-exponential otherwise, so scipy is optional.)

Model: lumped RC ladder down the arm. Each joint has a motor node (copper loss
in) tied to a housing structure (R2); structures chain joint-to-joint (R_link);
humerus/forearm structures shed heat through a fabric layer (R_stack) to ambient
(convective b*Area); wrist/hand are bare metal to ambient; the shoulder is
enclosed (no air path) and the J1 motor sinks to the battery/torso through
R_torso. Motor capacitance is time-varying: a sharp |dP/dt| transient briefly
collapses C toward the bare-winding mass and it recovers with time constant tau.

(c) Figure 2025 — Proprietary and Confidential.
================================================================================
"""
import numpy as np

try:
    from scipy.linalg import expm as _expm
except Exception:                                       # numpy-only fallback (scaling & squaring)
    def _expm(A):
        n = max(0, int(np.ceil(np.log2(max(np.sum(np.abs(A), axis=1).max(), 1e-12))))) + 1
        As = A / (2.0 ** n); E = np.eye(A.shape[0]); term = np.eye(A.shape[0])
        for k in range(1, 19):
            term = term @ As / k; E = E + term
        for _ in range(n):
            E = E @ E
        return E


# ============================================================================ #
#  FITTED CONFIGURATION  (edit here to retune / port to another arm build)
# ============================================================================ #
ACTS    = ['J1', 'J2', 'TWIST', 'ELBOW', 'ROLL', 'PITCH', 'YAW']         # shoulder -> hand
STRUCTS = ['shoulder', 'hum_u', 'hum_l', 'fore_u', 'fore_l', 'wrist', 'hand']
FABRICS = ['hum_u', 'hum_l', 'fore_u', 'fore_l']                          # fabric-covered structures

# (motor housing-structure, downstream output-structure) for each actuator
TOPO = {'J1': ('shoulder', 'shoulder'), 'J2': ('shoulder', 'hum_u'),
        'TWIST': ('hum_l', 'hum_u'), 'ELBOW': ('fore_u', 'hum_l'),
        'ROLL': ('fore_l', 'fore_u'), 'PITCH': ('wrist', 'fore_l'),
        'YAW': ('hand', 'wrist')}
ENCLOSED = {'shoulder': 'hum_u'}        # no air path; init inherits this neighbour's metal temp

# --- thermal capacitances [J/K] ------------------------------------------------
C_M    = np.array([108.747, 203.9, 203.9, 203.9, 114.16, 138.55, 155.12])   # full motor assembly
C_WIND = np.array([271.8675, 55.275, 55.275, 55.275, 26.0, 25.875, 26.0])   # bare-winding floor (adaptive)
C_S_FIX = {'shoulder': 676.0, 'hum_u': 222.3, 'hum_l': 778.1, 'fore_u': 1218.1,
           'fore_l': 399.7, 'wrist': 53.92, 'hand': 245.91}                 # structure metal
C_F_FIX = {'hum_u': 61.5, 'hum_l': 6.2, 'fore_u': 32.8, 'fore_l': 50.2}     # fabric

# --- geometry / fixed resistances ---------------------------------------------
R_STACK = {'hum_u': 1.749, 'hum_l': 0.227, 'fore_u': 1.319, 'fore_l': 1.913}  # structure<->fabric [K/W]
AREA = {'shoulder': 0.015875, 'hum_u': 0.031686, 'hum_l': 0.039904, 'fore_u': 0.026855,
        'fore_l': 0.027882, 'wrist': 0.015730, 'hand': 0.049}                 # convective area [m^2]

# --- FITTED parameters --------------------------------------------------------
R2     = {'J1': 1.0, 'J2': 1.4, 'TWIST': 1.4, 'ELBOW': 1.4,                   # motor->housing [K/W]
          'ROLL': 1.4, 'PITCH': 1.036, 'YAW': 1.059}
R_LINK = {'J1': 0.0, 'J2': 0.412, 'TWIST': 2.145, 'ELBOW': 15.0,             # housing->output [K/W]
          'ROLL': 0.068, 'PITCH': 1.563, 'YAW': 3.182}                       # (J1 unused: housing==output)
B_CONV = {'hum_u': 27.998, 'hum_l': 8.246, 'fore_u': 34.095, 'fore_l': 16.618,  # convection coeff [W/m^2K]
          'wrist': 6.0, 'hand': 6.0}                                          # surface->ambient = 1/(b*Area)
R_TORSO = 2.0                            # J1 motor -> battery/torso [K/W]
TORSO_DEFAULT = 40.0                     # fallback torso temp if none supplied [degC]

# --- heat sources -------------------------------------------------------------
QFET = 4.0                               # always-on drive/FET loss per actuator [W]
EXTRA_Q = {'YAW': 4.36}                  # hand low-voltage electronics [W]

# --- per-phase winding resistance at 20 C [ohm]  (copper power: 1.5*iq^2*R(T)) -
R20 = {'J1': 0.096, 'J2': 0.098, 'TWIST': 0.098, 'ELBOW': 0.098,
       'ROLL': 0.45, 'PITCH': 0.45, 'YAW': 0.45}

# --- adaptive motor-capacitance dynamics --------------------------------------
DPDT_THRESH = 5.0        # |dP/dt| [W/s] that fully collapses C to the winding floor
CADAPT_TAU  = 80.0       # recovery time constant back toward full assembly C [s]
ADAPT_C     = True       # set False to use constant full C_M


# ============================================================================ #
#  THE MODEL
# ============================================================================ #
class ArmThermalROM:
    """18-state RC thermal model of the F03 left arm. See module docstring."""

    def __init__(self):
        self.ACTS, self.STRUCTS, self.FABRICS = ACTS, STRUCTS, FABRICS
        self.NM, self.NS, self.NF = len(ACTS), len(STRUCTS), len(FABRICS)
        self.NX = self.NM + self.NS + self.NF                       # 7 + 7 + 4 = 18
        self.SIDX = {s: self.NM + k for k, s in enumerate(STRUCTS)}  # structure node index
        self.FIDX = {s: self.NM + self.NS + k for k, s in enumerate(FABRICS)}
        self.BARE = [s for s in STRUCTS if s not in FABRICS and s not in ENCLOSED]  # wrist, hand
        self.torso_node = ACTS.index('J1')

        self.C_M = np.asarray(C_M, float); self.C_WIND = np.asarray(C_WIND, float)
        self.C_FIX = np.array([C_S_FIX[s] for s in STRUCTS] + [C_F_FIX[s] for s in FABRICS])
        self.R20 = np.array([R20[a] for a in ACTS])
        self.extra_q = np.array([EXTRA_Q.get(a, 0.0) for a in ACTS])

        self.n_in = 2 * self.NM + 2          # u = [P(7), QF(7), Tamb, Ttorso]
        self._opcache = {}
        self._build()
        self.reset()

    # -------- network assembly (continuous-time G, Bq) -------------------------
    def _build(self):
        NM, NX = self.NM, self.NX
        G = np.zeros((NX, NX)); Bq = np.zeros((NX, self.n_in))

        def couple(n1, n2, R):                       # add a 1/R conductance between two nodes
            G[n1, n1] -= 1.0 / R
            if n2 == 'amb':
                Bq[n1, 2 * NM] += 1.0 / R            # ambient is an input column, not a state
            else:
                G[n1, n2] += 1.0 / R; G[n2, n1] += 1.0 / R; G[n2, n2] -= 1.0 / R

        for i, a in enumerate(ACTS):
            h = self.SIDX[TOPO[a][0]]; o = self.SIDX[TOPO[a][1]]
            couple(i, h, R2[a])                       # motor -> housing structure
            if o != h:
                couple(h, o, R_LINK[a])               # housing -> downstream structure
            Bq[h, NM + i] += 1.0                       # FET (+extra) load into housing structure
        for s in FABRICS:
            couple(self.SIDX[s], self.FIDX[s], R_STACK[s])             # structure <-> fabric
            couple(self.FIDX[s], 'amb', 1.0 / (B_CONV[s] * AREA[s]))   # fabric -> ambient
        for s in self.BARE:
            couple(self.SIDX[s], 'amb', 1.0 / (B_CONV[s] * AREA[s]))   # bare metal -> ambient
        # shoulder (enclosed): no ambient path; anchored only through its actuators
        G[self.torso_node, self.torso_node] -= 1.0 / R_TORSO          # J1 motor -> torso
        Bq[self.torso_node, 2 * NM + 1] += 1.0 / R_TORSO              # torso temp input column
        self.G, self.Bq = G, Bq

    def _mats(self, Cm):
        """continuous A, B for a given motor-capacitance vector Cm (struct/fabric C fixed)."""
        Cinv = 1.0 / np.concatenate([Cm, self.C_FIX])
        A = self.G * Cinv[:, None]; B = self.Bq * Cinv[:, None]
        B[:self.NM, :self.NM] = np.diag(Cinv[:self.NM])    # copper power P injected directly into motors
        return A, B

    def _discrete(self, Cm, dt):
        """zero-order-hold discrete operators (Ad, Bd), cached by (dt, quantized C)."""
        key = (round(dt, 6), tuple(np.round(Cm, 1)))
        op = self._opcache.get(key)
        if op is None:
            A, B = self._mats(Cm)
            Ad = _expm(A * dt)
            Bd = np.linalg.solve(A, (Ad - np.eye(self.NX))) @ B
            op = (Ad, Bd); self._opcache[key] = op
        return op

    # -------- copper loss ------------------------------------------------------
    def copper_power(self, iq, Tmotor):
        """real copper loss per actuator [W] = 1.5 * iq^2 * R20 * (234.5+T)/(234.5+20)."""
        iq = np.asarray(iq, float); Tmotor = np.asarray(Tmotor, float)
        return 1.5 * iq ** 2 * self.R20 * (234.5 + Tmotor) / 254.5

    # -------- adaptive motor capacitance (online recursion) --------------------
    def _cmotor(self, P, dt):
        """update the per-motor capacitance for this step's power P [W] (len NM)."""
        if not ADAPT_C:
            return self.C_M.copy()
        dP = np.abs(P - self._P_prev)
        ev = np.clip((dP / dt) / DPDT_THRESH, 0.0, 1.0)
        self._g = np.maximum(self._g * np.exp(-dt / CADAPT_TAU), ev)
        self._P_prev = P.copy()
        f = 1.0 - self._g
        return self.C_WIND + (self.C_M - self.C_WIND) * f

    def reset(self):
        """clear the adaptive-capacitance state (call before a fresh run)."""
        self._g = np.zeros(self.NM); self._P_prev = np.zeros(self.NM)

    # -------- initial state ----------------------------------------------------
    def init_state(self, Tmotor=None, Tamb=25.0, Tbatt=None):
        """seed the 18-state vector.

        Tmotor: array len NM of measured motor temps (NaN where a sensor is missing),
                or None (all missing). Missing motors / all structures are seeded from
                the mean of the available motor readings, else the battery temp, else Tamb.
        """
        self.reset()
        x = np.zeros(self.NX)
        Tbatt = TORSO_DEFAULT if Tbatt is None else Tbatt
        if Tmotor is None:
            Tmotor = np.full(self.NM, np.nan)
        Tmotor = np.asarray(Tmotor, float)
        good = [Tmotor[i] for i in range(self.NM) if np.isfinite(Tmotor[i])]
        seed = float(np.mean(good)) if good else float(Tbatt if Tbatt is not None else Tamb)
        for i in range(self.NM):
            x[i] = Tmotor[i] if np.isfinite(Tmotor[i]) else seed
        for s in STRUCTS:
            x[self.SIDX[s]] = seed
        for s in FABRICS:
            x[self.FIDX[s]] = seed
        return x

    # -------- one online step --------------------------------------------------
    def step(self, x, iq=None, Tamb=25.0, Tbatt=None, dt=1.0, clamp=None, power=None):
        """advance the state one tick and return (x_next).

        iq    : array len NM of q-axis currents [A]   (ignored if `power` is given)
        power : array len NM of copper loss [W]        (optional; overrides iq)
        Tamb  : ambient temperature [degC]
        Tbatt : battery/torso temperature [degC]       (defaults to TORSO_DEFAULT)
        clamp : dict {actuator_name_or_index: measured_T} to anchor healthy sensors.
                Anchored motors are forced to their reading; the rest stay predicted.

        Read the predicted motor temps as x_next[:7]; structures/fabrics follow.
        """
        x = np.asarray(x, float).copy()
        Tbatt = TORSO_DEFAULT if Tbatt is None else Tbatt
        P = np.asarray(power, float) if power is not None else self.copper_power(iq, x[:self.NM])
        Cm = self._cmotor(P, dt)
        Ad, Bd = self._discrete(Cm, dt)
        QF = np.full(self.NM, QFET) + self.extra_q
        u = np.concatenate([P, QF, [Tamb], [Tbatt]])
        x = Ad @ x + Bd @ u
        if clamp:
            for k, v in clamp.items():
                i = self.ACTS.index(k) if isinstance(k, str) else int(k)
                if v is not None and np.isfinite(v):
                    x[i] = v
        return x

    # -------- offline run over arrays -----------------------------------------
    def simulate(self, iq=None, Tamb=25.0, Tbatt=None, dt=1.0, x0=None, power=None, clamp_series=None):
        """integrate a whole logged run.

        iq    : (n, NM) currents  (or pass `power` (n, NM) instead)
        Tamb  : (n,) or scalar ambient ; Tbatt: (n,) or scalar torso
        x0    : initial 18-state (else seeded from the first motor reading / battery)
        clamp_series : optional (n, NM) array of measured motor temps; finite entries
                       anchor that motor that step (NaN = predict).  Use for validation
                       or to feed healthy thermistors while predicting dropped ones.
        Returns X of shape (n, 18).  Motor temps are X[:, :7].
        """
        src = np.asarray(power if power is not None else iq, float)
        n = len(src)
        Tamb = np.full(n, Tamb) if np.isscalar(Tamb) else np.asarray(Tamb, float)
        Tb = (np.full(n, TORSO_DEFAULT if Tbatt is None else Tbatt) if np.isscalar(Tbatt) or Tbatt is None
              else np.asarray(Tbatt, float))
        x = self.init_state(Tamb=Tamb[0], Tbatt=Tb[0]) if x0 is None else np.asarray(x0, float).copy()
        self.reset()
        X = np.zeros((n, self.NX)); X[0] = x
        for k in range(n - 1):
            clamp = None
            if clamp_series is not None:
                row = np.asarray(clamp_series[k], float)
                clamp = {i: row[i] for i in range(self.NM) if np.isfinite(row[i])}
            kw = dict(power=src[k]) if power is not None else dict(iq=src[k])
            x = self.step(x, Tamb=Tamb[k], Tbatt=Tb[k], dt=dt, clamp=clamp, **kw)
            X[k + 1] = x
        return X

    # -------- steady state -----------------------------------------------------
    def steady_state(self, power, Tamb, Tbatt=None):
        """solve all node temps at steady state for constant copper power [W] (len NM)."""
        NM = self.NM; Tbatt = TORSO_DEFAULT if Tbatt is None else Tbatt
        QF = np.full(NM, QFET) + self.extra_q
        rhs = self.Bq[:, NM:2 * NM] @ QF + self.Bq[:, 2 * NM] * Tamb + self.Bq[:, 2 * NM + 1] * Tbatt
        rhs[:NM] = rhs[:NM] + np.asarray(power, float)
        return np.linalg.solve(self.G, -rhs)


# ============================================================================ #
#  DEMO
# ============================================================================ #
if __name__ == '__main__':
    rom = ArmThermalROM()
    print('states=%d  motors=%s' % (rom.NX, rom.ACTS))

    # steady-state winding temps for a moderate hold (10 A on J2/ELBOW, idle elsewhere)
    iq = np.array([0, 10, 0, 8, 0, 0, 0.0])
    P = rom.copper_power(iq, np.full(rom.NM, 60.0))
    T = rom.steady_state(P, Tamb=25.0, Tbatt=40.0)
    print('steady motor T [C]:', dict(zip(rom.ACTS, np.round(T[:rom.NM], 1))))

    # 10-minute online transient from a 30 C cold start, 2 s ticks
    x = rom.init_state(Tmotor=np.full(rom.NM, 30.0), Tamb=25.0, Tbatt=40.0)
    for _ in range(300):
        x = rom.step(x, iq=iq, Tamb=25.0, Tbatt=40.0, dt=2.0)
    print('after 10 min  [C]:', dict(zip(rom.ACTS, np.round(x[:rom.NM], 1))))
