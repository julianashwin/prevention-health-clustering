"""Early mode check for a running multidim state-space fit.

The K=3 multidim SSM has at least two local optima on small samples.
Persistence is now class x channel; the check reads the physical channel (rho.k.1). In one,
classes 2 and 3 are separated by the anchor INTERCEPT; in the other their
intercepts sit pinned together against the ordering constraint and the
classes separate by SLOPE instead. Chains in different modes produce a large
rho R-hat with tiny within-chain spread.

This reads the live chain CSVs and reports, per chain, the quantities that
distinguish the modes -- so a bad run can be stopped hours in rather than
after twenty. Safe to run at any time; chains still in warmup are skipped.

    PYTHONPATH=src .venv/bin/python clustering/runs/multidim_mode_check.py \
        artifacts/multidim-health/theta-ssm-mort
"""

from __future__ import annotations

import glob
import sys

import numpy as np


def read(f):
    rows = [r for r in open(f) if not r.startswith("#")]
    if len(rows) < 30:
        return None, None
    hdr = rows[0].strip().split(",")
    w = len(hdr)
    dat = np.array([[float(x) for x in r.strip().split(",")]
                    for r in rows[1:] if len(r.strip().split(",")) == w])
    return hdr, dat


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    if not argv:
        print(__doc__)
        return 2
    files = sorted(glob.glob(argv[0].rstrip("/") + "/chains/*.csv")) \
        or sorted(glob.glob(argv[0].rstrip("/") + "/*.csv"))
    if not files:
        print("no chain csvs found")
        return 2
    print(f"{'chain':>6s} {'draws':>6s} {'rho1':>7s} {'rho2':>7s} {'rho3':>7s} "
          f"{'int2':>8s} {'int3':>8s} {'int3-int2':>10s} {'slope2':>8s}")
    keep = []
    for i, f in enumerate(files, 1):
        hdr, dat = read(f)
        if hdr is None:
            print(f"{i:>6d} {'--':>6s}   (warmup)")
            continue
        g = lambda n: dat[:, hdr.index(n)].mean()
        r = [g(f"rho.{k}.1") for k in (1, 2, 3)]  # physical-channel persistence
        i2, i3, s2 = g("coef.1.2.1"), g("coef.1.3.1"), g("coef.1.2.2")
        keep.append(r + [i3 - i2])
        print(f"{i:>6d} {len(dat):>6d} " + " ".join(f"{x:7.4f}" for x in r)
              + f" {i2:8.4f} {i3:8.4f} {i3 - i2:10.4f} {s2:8.4f}")
    if len(keep) >= 2:
        a = np.array(keep)
        spread = a[:, 1:3].max(axis=0) - a[:, 1:3].min(axis=0)
        gapmin, gapmax = a[:, 3].min(), a[:, 3].max()
        print(f"\n  rho[2]/rho[3] spread across chains: "
              f"{spread[0]:.4f} / {spread[1]:.4f}")
        print(f"  intercept gap (int3-int2) ranges {gapmin:.4f} to {gapmax:.4f}")
        bad = spread.max() > 0.01 or gapmin < 0.05
        print("\n  " + ("MODE SPLIT: chains are in different optima -- "
                        "stop the run and rethink"
                        if bad else
                        "chains agree; the fit is in one mode"))
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
