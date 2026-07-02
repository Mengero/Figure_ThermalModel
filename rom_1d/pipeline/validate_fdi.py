"""Offline validation of the model-based FDI + virtual-sensor layer (core/observer.py).

Modes:
  --inject SENSOR TYPE T0 T1   inject a synthetic fault into a CLEAN case; report detection
                               latency, false alarms on other sensors, and virtual-sensor error
                               (effective vs withheld true) over the fault window.
  --real                       run on a case with a REAL fault (e.g. c_1076 dead YAW).
  --sweep K                    drop k=1..K motors (ALL combinations) from boot; report the
                               worst/best virtual-sensor error vs #dropouts -> the usability limit.

Cases: arm wide cases by --case (data/arm/data_<case>.csv), or a field long-format file by --csv.
"""
import argparse, itertools, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import _common as C
sys.path.insert(0, C.ROOT)
from core.observer import ThermalObserver          # noqa: E402
import _faults                                      # noqa: E402

USABLE = 15.0    # +-15 C virtual-sensor usability bound (Function 3)
SH = {'left_shoulder_j1': 'J1', 'left_shoulder_j2': 'J2', 'left_upper_arm_twist': 'TWIST',
      'left_elbow': 'ELBOW', 'left_wrist_roll': 'ROLL', 'left_wrist_pitch': 'PITCH', 'left_wrist_yaw': 'YAW'}


def load_wide(M, case, csv, dt, amb_offset):
    """return a wide df (t_s, T_amb, [T_torso], P_<act>, Tm_<act>) from an arm case or a field log."""
    raw = pd.read_csv(csv) if csv else C.load(M.cfg.name, case)
    if 'actuator_name' not in raw.columns and 'J' not in raw.columns:
        # already wide (arm case): resample to the requested dt so the ZOH step matches the grid
        ts = raw['t_s'].to_numpy()
        if len(ts) > 1 and abs(np.median(np.diff(ts)) - dt) > 1e-6:
            grid = np.arange(ts[0], ts[-1] + 1e-6, dt)
            out = pd.DataFrame({'t_s': grid})
            for c in raw.columns:
                if c != 't_s':
                    out[c] = np.interp(grid, ts, raw[c].to_numpy())
            return out
        return raw
    raw['t'] = pd.to_datetime(raw['timestamp_utc']); raw = raw.sort_values('t')
    raw['J'] = raw['J'] if 'J' in raw.columns else raw['actuator_name'].map(SH)
    t0 = raw['t'].min(); raw['s'] = (raw['t'] - t0).dt.total_seconds()
    grid = np.arange(0, raw['s'].max() + 1e-6, dt)
    df = pd.DataFrame({'t_s': grid})
    R20 = M.cfg.R20
    for J in M.ACTS:
        g = raw[raw['J'] == J].sort_values('s')
        if g.empty:
            df['Tm_' + J] = np.nan; df['P_' + J] = 0.0; continue
        Tm = np.interp(grid, g['s'], g['Tmotor_degC'])
        if 'real_power_W' in g.columns:
            P = np.interp(grid, g['s'], g['real_power_W'])
        else:                                        # raw long: power_W = 3/4 iq^2 -> 1.5 iq^2 R(T)
            P = 2.0 * np.interp(grid, g['s'], g['power_W']) * R20[J] * (234.5 + Tm) / 254.5
        df['Tm_' + J] = Tm; df['P_' + J] = P
    ambcol = 'Tamb_filled' if 'Tamb_filled' in raw.columns else 'Tambient_degC'
    a = raw.dropna(subset=[ambcol]).sort_values('s')
    df['T_amb'] = (np.interp(grid, a['s'], a[ambcol]) + amb_offset) if len(a) else 25.0
    bcol = 'Tbatt_max_filled' if 'Tbatt_max_filled' in raw.columns else ('Tbatt_max_degC' if 'Tbatt_max_degC' in raw.columns else None)
    if bcol:
        b = raw.dropna(subset=[bcol]).sort_values('s')
        if len(b):
            df['T_torso'] = np.interp(grid, b['s'], b[bcol])
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limb', default='arm')
    ap.add_argument('--case', default='0616cont')
    ap.add_argument('--csv', default=None)
    ap.add_argument('--dt', type=float, default=2.0)
    ap.add_argument('--amb-offset', type=float, default=0.0)
    ap.add_argument('--inject', nargs=4, metavar=('SENSOR', 'TYPE', 'T0', 'T1'))
    ap.add_argument('--real', action='store_true')
    ap.add_argument('--sweep', type=int, default=0)
    a = ap.parse_args()
    M = C.rom(a.limb); M.build(C.load_params(a.limb))
    df = load_wide(M, a.case, a.csv, a.dt, a.amb_offset)
    tag = a.case if not a.csv else os.path.splitext(os.path.basename(a.csv))[0]

    if a.inject:
        sensor, ftype, t0, t1 = a.inject[0], a.inject[1], float(a.inject[2]), float(a.inject[3])
        fdf, true, m = _faults.inject(df, sensor, ftype, t0, t1)
        r = ThermalObserver(M).run(fdf, a.dt); t = r['t']; i = M.ACTS.index(sensor)
        fl = r['flags'][:, i]
        det = np.where(fl & (t >= t0))[0]
        lat = (t[det[0]] - t0) * 60 if len(det) else np.nan
        false = {M.ACTS[j]: int(r['flags'][:, j].sum()) for j in range(M.NM) if j != i and r['flags'][:, j].any()}
        we = r['eff'][m, i] - true[m]                 # virtual vs withheld true, over fault window
        print('INJECT %s %s [%g,%g]min on %s:' % (sensor, ftype, t0, t1, tag))
        print('  detection latency = %s' % ('%.0f s' % lat if np.isfinite(lat) else 'NOT DETECTED'))
        print('  false alarms on other sensors: %s' % (false or 'none'))
        print('  virtual-sensor err over fault: RMSE %.2f  max %.2f C' % (np.sqrt(np.nanmean(we**2)), np.nanmax(np.abs(we))))
        fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax[0].plot(t, true, 'g-', lw=1.6, label='true (withheld)')
        ax[0].plot(t, fdf['Tm_' + sensor], 'r.', ms=2, label='faulted reading')
        ax[0].plot(t, r['eff'][:, i], 'b--', lw=1.4, label='effective (model when faulted)')
        ax[0].axvspan(t0, t1, color='k', alpha=.06); ax[0].set_ylabel('%s T [°C]' % sensor); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
        ax[1].plot(t, r['innov'][:, i], lw=1, label='innovation'); ax[1].fill_between(t, 0, 1, where=fl, transform=ax[1].get_xaxis_transform(), color='r', alpha=.15, label='FAULT flag')
        ax[1].axhline(0, color='k', lw=.5); ax[1].set_ylabel('innovation [°C]'); ax[1].set_xlabel('time [min]'); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
        ax[0].set_title('FDI inject: %s %s on %s (lat %s, virt RMSE %.1f C)' % (sensor, ftype, tag, '%.0fs' % lat if np.isfinite(lat) else 'miss', np.sqrt(np.nanmean(we**2))))
        out = C.fig_path('fdi_%s_%s_%s.png' % (tag, sensor, ftype), a.limb); plt.tight_layout(); plt.savefig(out, dpi=600); print('  saved', out)

    elif a.real:
        r = ThermalObserver(M).run(df, a.dt); t = r['t']
        print('REAL-fault run on %s:' % tag)
        for j, J in enumerate(M.ACTS):
            if r['flags'][:, j].any():
                k = np.argmax(r['flags'][:, j]); print('  %s flagged at %.1f min (cause=%s)' % (J, t[k], r['cause'][k][j]))
        col = dict(zip(M.ACTS, plt.cm.tab10.colors)); fig, ax = plt.subplots(figsize=(12, 6))
        for j, J in enumerate(M.ACTS):
            ax.plot(t, r['eff'][:, j], ('--' if r['flags'][:, j].any() else '-'), lw=1.4, color=col[J],
                    label=J + (' (virtual)' if r['flags'][:, j].any() else ''))
        if 'T_torso' in df: ax.plot(t, df['T_torso'], 'k:', lw=1, label='battery/torso')
        ax.set_xlabel('time [min]'); ax.set_ylabel('T [°C]'); ax.set_title('FDI on %s — effective temps (virtual where flagged)' % tag)
        ax.grid(alpha=.3); ax.legend(fontsize=8, ncol=4)
        out = C.fig_path('fdi_real_%s.png' % tag, a.limb); plt.tight_layout(); plt.savefig(out, dpi=600); print('  saved', out)

    elif a.sweep:
        truth = {J: df['Tm_' + J].to_numpy() for J in M.ACTS}
        rows = []
        for k in range(1, a.sweep + 1):
            best = (1e9, None); worst = (-1, None)
            for combo in itertools.combinations(range(M.NM), k):
                r = ThermalObserver(M, forced_drop=combo).run(df, a.dt)
                errs = [np.nanmax(np.abs(r['eff'][:, j] - truth[M.ACTS[j]])) for j in combo]
                me = max(errs)
                if me < best[0]: best = (me, combo)
                if me > worst[0]: worst = (me, combo)
            rows.append((k, best[0], worst[0], worst[1]))
            print('k=%d dropped: best-case maxerr=%.1f  worst-case maxerr=%.1f  (worst combo=%s)'
                  % (k, best[0], worst[0], [M.ACTS[j] for j in worst[1]]))
        ks = [r[0] for r in rows]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(ks, [r[1] for r in rows], 'g-o', label='best-case combo')
        ax.plot(ks, [r[2] for r in rows], 'r-o', label='worst-case combo')
        ax.axhline(USABLE, color='k', ls='--', label='±%g C usable' % USABLE)
        ax.set_xlabel('# thermistors dropped'); ax.set_ylabel('max virtual-sensor error [°C]')
        ax.set_title('%s FDI dropout limit (%s)' % (a.limb, tag)); ax.grid(alpha=.3); ax.legend()
        out = C.fig_path('fdi_sweep_%s.png' % tag, a.limb); plt.tight_layout(); plt.savefig(out, dpi=600); print('saved', out)
        ok = [r[0] for r in rows if r[2] <= USABLE]
        print('=> worst-case stays within ±%g C up to %d simultaneous dropouts' % (USABLE, max(ok) if ok else 0))


if __name__ == '__main__':
    main()
