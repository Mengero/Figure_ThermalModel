"""Limb registry. Add a new limb by writing <name>.py with a `CONFIG = LimbConfig(...)`
and listing it here — the engine and pipeline are reused unchanged."""
import importlib

_LIMBS = {'arm', 'leg'}


def load(name):
    """return the LimbConfig for a limb name (e.g. 'arm', 'leg')."""
    if name not in _LIMBS:
        raise ValueError(f"unknown limb '{name}'; known: {sorted(_LIMBS)}")
    return importlib.import_module(f'limbs.{name}').CONFIG
