#!/usr/bin/env python
"""1D thermal ROM — one-command pipeline, parametrized by limb.

Usage (inside conda env thermal_sim, from rom_1d/):
    python main.py --limb arm                 # fit + train plot + test plot
    python main.py --limb arm all             # everything (incl. hi19, energy, current limit)
    python main.py --limb arm fit test        # pick steps
    python main.py --limb leg fit             # once leg config + data are filled in

Steps:  fit  trainplot  testplot  hi19  energy  climit
(the same pipeline scripts serve every limb; only the LimbConfig changes.)
"""
import argparse, subprocess, sys, os

PIPE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline')
STEPS = {
    'fit':       ['fit.py'],
    'trainplot': ['plot_fit.py', '--split', 'train'],
    'testplot':  ['plot_fit.py', '--split', 'test'],
    'hi19':      ['plot_hi19.py'],
    'conttest':  ['test_cont.py', '--case', '0616cont'],   # continuous run: motors+structures held-out test
    'energy':    ['energy.py'],
    'climit':    ['current_limit.py'],
}
ORDER = ['fit', 'trainplot', 'testplot', 'hi19', 'conttest', 'energy', 'climit']
DEFAULT = ['fit', 'trainplot', 'testplot']


def run(step, limb):
    script, *extra = STEPS[step]
    print(f"\n{'='*60}\n>>> {step}  ({script} --limb {limb} {' '.join(extra)})\n{'='*60}")
    r = subprocess.run([sys.executable, os.path.join(PIPE, script), '--limb', limb] + extra)
    if r.returncode != 0:
        print(f"!! step '{step}' failed (exit {r.returncode})"); sys.exit(r.returncode)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--limb', default='arm')
    ap.add_argument('steps', nargs='*', help="steps or 'all'; default: fit trainplot testplot")
    a = ap.parse_args()
    if not a.steps:
        chosen = DEFAULT
    elif a.steps == ['all']:
        chosen = ORDER
    else:
        bad = [s for s in a.steps if s not in STEPS]
        if bad:
            print(f"unknown step(s) {bad}; valid: {list(STEPS)} or 'all'"); sys.exit(1)
        chosen = [s for s in ORDER if s in a.steps]
    for s in chosen:
        run(s, a.limb)
    print(f"\nDone ({a.limb}). Outputs in fits/ and figures/.")
