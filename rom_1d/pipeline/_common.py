"""Shared plumbing for the limb pipeline scripts: paths, ROM construction, IO."""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['legend.frameon'] = False   # aligned style: no legend box

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # rom_1d/
sys.path.insert(0, ROOT)

from core import ThermalROM           # noqa: E402
import limbs                          # noqa: E402

DATA_DIR = os.path.join(ROOT, 'data')
FITS_DIR = os.path.join(ROOT, 'fits')
FIG_DIR = os.path.join(ROOT, 'figures')


def rom(limb):
    """build a ThermalROM for a limb name."""
    return ThermalROM(limbs.load(limb))


def data_path(limb, tag):
    return os.path.join(DATA_DIR, limb, f'data_{tag}.csv')


def load(limb, tag):
    return pd.read_csv(data_path(limb, tag))


def fit_path(limb):
    return os.path.join(FITS_DIR, f'{limb}_fitted_params.json')


def load_params(limb):
    return np.array(json.load(open(fit_path(limb)))['p'])


def save_params(limb, obj):
    os.makedirs(FITS_DIR, exist_ok=True)
    json.dump(obj, open(fit_path(limb), 'w'), indent=1, default=float)


def fig_path(name, sub=None):
    d = FIG_DIR if sub is None else os.path.join(FIG_DIR, sub)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)
