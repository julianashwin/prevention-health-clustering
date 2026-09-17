"""Rebuild a run summary from chain CSVs when sampling was cut short.

A long fit that is interrupted -- a reboot, a killed session -- leaves its
chain CSVs on disk with every draw written up to that moment. Those draws are
ordinary post-warmup draws: adaptation has already finished, so they are a
valid, merely smaller, posterior sample. CmdStan cannot resume from where it
stopped, so the choice is between using what was written and paying for the
whole fit again.

This reproduces what run_fit.py writes at the end of a successful fit, reading
the CSVs directly instead of through cmdstanpy (which refuses files whose
sampling is incomplete). Two details matter:

  * the final row of each file is usually truncated mid-write, and is dropped
    on a column-count check rather than silently parsed short;
  * chains stop at different draw counts, so every chain is cut to the
    shortest one before anything is computed. Split R-hat needs equal lengths,
    and using different lengths for the means than for R-hat would make the
    two disagree about which sample they describe.

The summary it writes carries salvaged: true and the per-chain draw counts, so
the shortfall against the planned sampling is visible to anything reading it.

Usage:
  python clustering/salvage_partial_fit.py --dir artifacts/k5/physfunc-ssm-k5 \
      --model physgrm-func-ssm-k5 [--planned-sampling 1000] [--write]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from prevention_health_clustering.models.registry import REGISTRY  # noqa: E402


def split_rhat(x: np.ndarray) -> float:
    """Classic split-R-hat over (chains, draws); no rank-normalisation.

    Identical to run_fit.split_rhat, duplicated so that salvaging a fit does
    not import the module that launches one.
    """
    c, d = x.shape
    half = d // 2
    parts = np.concatenate([x[:, :half], x[:, half:2 * half]], axis=0)
    m = parts.mean(axis=1)
    w = parts.var(axis=1, ddof=1).mean()
    b = half * m.var(ddof=1)
    var_plus = (half - 1) / half * w + b / half
    return float(np.sqrt(var_plus / w)) if w > 0 else np.inf


def read_chain(path: Path) -> tuple[list[str], np.ndarray, int]:
    """Parse one CmdStan CSV, dropping any row that is not full width."""
    rows = [ln for ln in path.read_text().splitlines() if not ln.startswith("#")]
    header = rows[0].split(",")
    width = len(header)
    good = [r.split(",") for r in rows[1:] if r.count(",") + 1 == width]
    dropped = len(rows) - 1 - len(good)
    return header, np.array(good, dtype=float), dropped


def elapsed_hours(log: Path) -> float | None:
    """Hours the run had been going when it stopped, from the progress bars.

    Not a completed-fit wall time -- the fit never completed -- but the honest
    analogue of it, and downstream summaries expect the field to be there.
    """
    if not log.exists():
        return None
    times = re.findall(r"\[(\d+):(\d\d):(\d\d)<", log.read_text())
    if not times:
        return None
    return max(int(h) + int(m) / 60 + int(s) / 3600 for h, m, s in times)


def bracket_to_dot(name: str) -> str:
    return name.replace("[", ".").replace("]", "").replace(",", ".")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True,
                    help="fit output directory containing chains/")
    ap.add_argument("--model", required=True, help="registry key")
    ap.add_argument("--planned-sampling", type=int, default=None,
                    help="draws per chain the run was launched with")
    ap.add_argument("--write", action="store_true",
                    help="write run_summary.json (otherwise print only)")
    args = ap.parse_args(argv)

    spec = REGISTRY[args.model]
    k, c = spec.n_classes, spec.n_channels
    ar_mode = getattr(spec, "ar_mode", 0)

    files = sorted((args.dir / "chains").glob("*_[0-9].csv"))
    if not files:
        print(f"no chain CSVs under {args.dir / 'chains'}", file=sys.stderr)
        return 1

    header, chains, drops = None, [], []
    for f in files:
        h, arr, d = read_chain(f)
        header = header or h
        chains.append(arr)
        drops.append(d)
        print(f"  {f.name}: {len(arr):,} draws ({d} truncated row dropped)")

    n = min(len(a) for a in chains)
    per_chain = [len(a) for a in chains]
    print(f"\n  cut to the shortest chain: {n:,} draws x {len(chains)} chains "
          f"= {n * len(chains):,} total"
          + (f" (planned {args.planned_sampling * len(chains):,})"
             if args.planned_sampling else ""))

    idx = {name: i for i, name in enumerate(header)}
    rho_names = [f"rho[{i}]" for i in range(1, k + 1)] if ar_mode != 0 else []
    meas_names = ([f"sigma_meas[{ci}]" for ci in range(1, c + 1)]
                  if ar_mode == 2 else [])
    names = ([f"theta[{i}]" for i in range(1, k + 1)]
             + [f"coef[{ci},{i},{p}]" for ci in range(1, c + 1)
                for i in range(1, k + 1) for p in range(1, 4)]
             + [f"sigma[1,{ci}]" for ci in range(1, c + 1)]
             + rho_names + meas_names + ["lp__"])

    summary, worst_rhat, worst_name = {}, 0.0, ""
    for name in names:
        col = np.stack([a[:n, idx[bracket_to_dot(name)]] for a in chains])
        r = split_rhat(col)
        summary[name] = {"mean": float(col.mean()), "sd": float(col.std()),
                         "rhat": r}
        if name != "lp__" and r > worst_rhat:
            worst_rhat, worst_name = r, name

    print(f"\n  max structural R-hat {worst_rhat:.4f} ({worst_name})")
    print("  theta " + " ".join(
        f"{summary[f'theta[{i}]']['mean']:.3f}" for i in range(1, k + 1)))
    if ar_mode != 0:
        print("  rho " + " ".join(f"{summary[nm]['mean']:.3f}" for nm in rho_names))
    if ar_mode == 2:
        print("  sigma_meas " + " ".join(
            f"{summary[nm]['mean']:.3f}" for nm in meas_names))
        sig = summary["sigma[1,1]"]["mean"]
        mea = summary[meas_names[0]]["mean"]
        shares = [(sig ** 2 / (1 - summary[nm]["mean"] ** 2))
                  / (sig ** 2 / (1 - summary[nm]["mean"] ** 2) + mea ** 2)
                  for nm in rho_names]
        print("  signal share " + " ".join(f"{x:.3f}" for x in shares))

    # Means on every draw written, as a check that cutting to the shortest
    # chain did not move them.
    drift = 0.0
    for name in names:
        full = np.concatenate([a[:, idx[bracket_to_dot(name)]] for a in chains])
        cut = summary[name]["mean"]
        if summary[name]["sd"] > 0:
            drift = max(drift, abs(full.mean() - cut) / summary[name]["sd"])
    print(f"  largest mean shift from using all {sum(per_chain):,} draws "
          f"instead: {drift:.3f} posterior sd")

    data = json.loads((args.dir / "stan_data.json").read_text())
    result = {
        "model": args.model,
        "n_person": int(data["N_person"]), "n_obs": int(data["N_obs"]),
        "held_rows": int(np.sum(data.get("heldout", 0))),
        "chains": len(chains), "sampling": n,
        "wall_hours": elapsed_hours(args.dir / "run.log"),
        "salvaged": True,
        "planned_sampling": args.planned_sampling,
        "draws_per_chain_on_disk": per_chain,
        "truncated_rows_dropped": drops,
        "max_structural_rhat": worst_rhat, "worst_param": worst_name,
        "params": summary,
    }
    if args.write:
        out = args.dir / "run_summary.json"
        out.write_text(json.dumps(result, indent=1))
        print(f"\n  wrote {out}")
    ok = worst_rhat < 1.05
    print(f"\n[{args.model}] {'CONVERGED' if ok else 'R-HAT FLAG'} "
          f"on {n * len(chains):,} salvaged draws")
    return 0


if __name__ == "__main__":
    sys.exit(main())
