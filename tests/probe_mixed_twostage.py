"""Stage-2 validation: mode-informed inits + metric + step size, same data.

Pass requires BOTH: fast warmup (the metric lever) AND landing in the good
mode the baseline found (lp ~ -92,976), with chains agreeing. The plain
metric lever failed the second condition (modes at -93,685 to -94,449).

Caveat this test cannot cover: at production, stage 1 is a SUBSAMPLE and
stage 2 the full roster, so inits sit near but not at the full-data mode.
"""
import glob, sys, time
import numpy as np
import pandas as pd
from prevention_health_clustering.models.registry import get_model
from prevention_health_clustering.runner.fit import compile_model
from prevention_health_clustering.runner.mixed import build_mixed_payload, write_mixed_data

BASELINE_LP = -92976.0
K, C, P = 3, 2, 3

# --- posterior means from the converged baseline chains ---
frames = []
for f in sorted(glob.glob("artifacts/mixed-probe/n2000/chains/*_[0-9].csv")):
    hdr, rows = None, []
    for line in open(f):
        if line.startswith("#"): continue
        if hdr is None: hdr = line.strip().split(","); continue
        rows.append(line.strip().split(","))
    frames.append(pd.DataFrame(np.asarray(rows, dtype=float), columns=hdr))
draws = pd.concat(frames, ignore_index=True)
m = draws.mean()

def arr(name, *shape):
    if len(shape) == 1:
        return np.array([m[f"{name}.{i+1}"] for i in range(shape[0])])
    if len(shape) == 2:
        return np.array([[m[f"{name}.{i+1}.{j+1}"] for j in range(shape[1])]
                         for i in range(shape[0])])
    return np.array([[[m[f"{name}.{i+1}.{j+1}.{l+1}"] for l in range(shape[2])]
                      for j in range(shape[1])] for i in range(shape[0])])

mode = {
    "theta": arr("theta", K),
    "anchor_intercept": np.sort(arr("anchor_intercept", K)),
    "other_intercept": arr("other_intercept", K, C - 1),
    "slope_gauss": arr("slope_gauss", C, K, P - 1),
    "sigma_gauss": arr("sigma_gauss", C),
    "coef_srh_raw": arr("coef_srh_raw", K, P),
    "cut_srh_gap": arr("cut_srh_gap", 3),
    "coef_chronic_raw": arr("coef_chronic_raw", K, P),
    "phi_chronic": float(m["phi_chronic"]),
    "coef_adl_zero_raw": arr("coef_adl_zero_raw", K, P),
    "coef_adl_count_raw": arr("coef_adl_count_raw", K, P),
    "phi_adl": float(m["phi_adl"]),
}
print(f"baseline mode: lp target {BASELINE_LP:.0f}, "
      f"theta {np.round(mode['theta'], 3).tolist()}")

rng = np.random.default_rng(20260825)
inits = []
for _ in range(4):
    j = lambda x, s=0.05: np.asarray(x) + rng.normal(0, s, np.shape(x))  # noqa: E731
    init = {
        "theta": (lambda v: (v / v.sum()).tolist())(
            np.maximum(j(mode["theta"], 0.01), 1e-3)),
        "anchor_intercept": np.sort(j(mode["anchor_intercept"])).tolist(),
        "other_intercept": j(mode["other_intercept"]).tolist(),
        "slope_gauss": j(mode["slope_gauss"], 0.02).tolist(),
        "sigma_gauss": np.maximum(j(mode["sigma_gauss"], 0.02), 0.06).tolist(),
        "coef_srh_raw": j(mode["coef_srh_raw"]).tolist(),
        "cut_srh_gap": np.maximum(j(mode["cut_srh_gap"], 0.02), 0.05).tolist(),
        "coef_chronic_raw": j(mode["coef_chronic_raw"]).tolist(),
        "phi_chronic": max(float(mode["phi_chronic"] + rng.normal(0, 0.5)), 0.2),
        "coef_adl_zero_raw": j(mode["coef_adl_zero_raw"]).tolist(),
        "coef_adl_count_raw": j(mode["coef_adl_count_raw"]).tolist(),
        "phi_adl": max(float(mode["phi_adl"] + rng.normal(0, 0.3)), 0.2),
    }
    inits.append(init)

spec = get_model("mixed-health")
full = pd.read_parquet("/tmp/phc_mixed_frame_full.parquet")
keep = full["pidp"].drop_duplicates().sample(2000, random_state=20260825)
payload = build_mixed_payload(spec, full[full["pidp"].isin(keep)])
model = compile_model(spec.stan_file)
data_path = write_mixed_data(payload, __import__("pathlib").Path("artifacts/mixed-twostage/n2000"))

start = time.time()
fit = model.sample(
    data=str(data_path), inits=inits,
    chains=4, iter_warmup=200, iter_sampling=200,
    adapt_delta=spec.adapt_delta, max_treedepth=spec.max_treedepth,
    seed=spec.seed, threads_per_chain=3,
    inv_metric="/tmp/phc_mixed_metric.json", step_size=0.065,
    output_dir="artifacts/mixed-twostage/n2000/chains", show_progress=False,
)
wall = time.time() - start

for path in fit.runset.csv_files:
    warm = samp = 0.0
    for line in open(path):
        if "(Warm-up)" in line:
            warm = float(line.replace("#", "").split()[2])
        elif "(Sampling)" in line and "seconds" in line:
            samp = float(line.replace("#", "").split()[0])
    print(f"chain: warmup {warm:6.1f}s ({warm/200:4.2f} s/it)  "
          f"sampling {samp:6.1f}s ({samp/200:4.2f} s/it)")

d2 = fit.draws_pd()
lp_by_chain = d2.groupby("chain__")["lp__"].mean() if "chain__" in d2 else None
rhat = float(fit.summary()["R_hat"].dropna().max())
lp_mean = float(d2["lp__"].mean())
theta = [float(d2[f"theta[{i}]"].mean()) for i in (1, 2, 3)]
print(f"\nwall {wall:.1f}s   max R-hat {rhat:.4f}")
print(f"lp mean {lp_mean:.1f}  (baseline mode {BASELINE_LP:.1f}, "
      f"gap {lp_mean - BASELINE_LP:+.1f})")
print(f"theta {np.round(theta, 4).tolist()}")

ok = rhat < 1.05 and abs(lp_mean - BASELINE_LP) < 60
print("\nTWO-STAGE", "VALIDATED" if ok else "REFUTED")
sys.exit(0 if ok else 1)
