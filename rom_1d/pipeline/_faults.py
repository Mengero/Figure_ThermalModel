"""Synthetic thermistor-fault injection for offline FDI validation.

Operates on a WIDE df (columns t_s, Tm_<act>, ...). Returns a copy with the chosen sensor
corrupted over [t0,t1] minutes, plus the untouched TRUE series for scoring.

Fault types (per the locked spec):
  dead   - reading jumps to a fixed dead value (default -72 C, like the real c_1076 YAW)
  nan    - reading goes NaN (sensor / bus dropout)
  oor    - reading jumps out of range (default 200 C)
  stuck  - reading freezes at its value at fault onset (== frozen)
"""
import numpy as np

DEAD_VALUE = -72.0
OOR_VALUE = 200.0


def inject(df, sensor, ftype, t0_min, t1_min):
    df = df.copy()
    col = 'Tm_' + sensor
    t = df['t_s'].to_numpy() / 60.0
    true = df[col].to_numpy().copy()
    m = (t >= t0_min) & (t <= t1_min)
    if not m.any():
        raise ValueError('fault window [%g,%g] min has no samples' % (t0_min, t1_min))
    onset = np.argmax(m)                       # first faulted index
    v = df[col].to_numpy().astype(float)
    if ftype == 'dead':
        v[m] = DEAD_VALUE
    elif ftype == 'nan':
        v[m] = np.nan
    elif ftype == 'oor':
        v[m] = OOR_VALUE
    elif ftype in ('stuck', 'frozen'):
        v[m] = true[onset]                     # freeze at the value at fault onset
    else:
        raise ValueError('unknown fault type %r' % ftype)
    df[col] = v
    return df, true, m
