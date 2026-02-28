"""
experiment_recovery_time_ws.py — Batch scanner for cascade recovery-time
experiments on Watts–Strogatz networks, with per-(K,q,alpha) npz caching.

Supports 3D sweep over (K, q, alpha) where the effective failure threshold
is theta_max = alpha * theta_base.

Usage examples:

    # Full 3D scan: K=4,6,8, 21 q-points, 21 alpha-points, R=20
    python experiment_recovery_time_ws.py --K_list 4,6,8 --q_points 21 \
        --alpha_points 21 --R 20 --t_max 120

    # Custom alpha range
    python experiment_recovery_time_ws.py --K_list 4,6,8 --q_points 11 \
        --alpha_min 0.3 --alpha_max 3.0 --alpha_points 15 --R 10

    # Explicit alpha values
    python experiment_recovery_time_ws.py --K_list 8 --q_list 0.15 \
        --alpha_list 0.5,1.0,1.5,2.0 --R 5

    # 2D scan (single alpha = legacy mode)
    python experiment_recovery_time_ws.py --K_list 4,6,8 --q_points 21 \
        --alpha_list 1.0 --R 20 --t_max 120

    # Save example timeseries for first 2 realizations
    python experiment_recovery_time_ws.py --K_list 4 --q_list 0.0 \
        --alpha_list 1.0 --R 10 --save_example_ts 2

    # Exponential repair, random shock
    python experiment_recovery_time_ws.py --K_list 4,6 --q_points 11 \
        --alpha_points 11 --R 10 \
        --shock_mode random --repair_dist exponential --repair_mean 3.0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cascade_utils import simulate_cascade_recovery_ws

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ====================================================================
# 1. Meta-parameter dict (for cache validation)
# ====================================================================

_META_KEYS = [
    "R", "dt", "t_max", "t_shock", "shock_mode", "kappa",
    "theta_base", "alpha", "fail_duration", "check_dt",
    "repair_mean", "repair_dist", "retry_delay", "max_retries",
    "eps", "hold_time", "N", "gen_ratio", "Pmax", "base_seed",
]


def _build_meta(args: dict, alpha: float) -> dict:
    """Extract the subset of parameters that define a cache-valid run."""
    meta = {k: args[k] for k in _META_KEYS if k != "alpha"}
    meta["alpha"] = alpha
    return meta


def _meta_match(stored: dict, current: dict) -> bool:
    """Check whether stored meta matches current parameters exactly."""
    for k in _META_KEYS:
        sv = stored.get(k)
        cv = current.get(k)
        # Compare floats with tolerance, others by equality
        if isinstance(cv, float):
            if not np.isclose(sv, cv, rtol=1e-9):
                return False
        else:
            if sv != cv:
                return False
    return True


# ====================================================================
# 2. Cache file naming & I/O
# ====================================================================

def _cache_filename(K: int, q: float, alpha: float, R: int, base_seed: int) -> str:
    """Deterministic cache filename for a (K, q, alpha) point."""
    return f"ws_K{K}_q{q:.3f}_a{alpha:.3f}_R{R}_seed{base_seed}.npz"


def _load_cache(path: Path, meta: dict) -> dict | None:
    """Load and validate a cached npz file. Returns dict or None."""
    if not path.exists():
        return None
    try:
        data = dict(np.load(path, allow_pickle=True))
        stored_meta = json.loads(str(data["meta"]))
        if _meta_match(stored_meta, meta):
            return data
    except Exception:
        pass
    return None


def _save_cache(
    path: Path,
    meta: dict,
    seeds: np.ndarray,
    T_rec: np.ndarray,
    A_res: np.ndarray,
    E_lost_max: np.ndarray,
    S_min: np.ndarray,
    t_S_min: np.ndarray,
    unrecovered: np.ndarray,
    example_ts: dict | None = None,
) -> None:
    """Save metrics arrays + meta to a compressed npz."""
    save_dict = {
        "meta": json.dumps(meta),
        "seeds": seeds,
        "T_rec": T_rec,
        "A_res": A_res,
        "E_lost_max": E_lost_max,
        "S_min": S_min,
        "t_S_min": t_S_min,
        "unrecovered": unrecovered,
    }
    # Optional example timeseries
    if example_ts:
        for key, val in example_ts.items():
            save_dict[f"ts_{key}"] = val

    np.savez_compressed(path, **save_dict)


# ====================================================================
# 3. Run one (K, q, alpha) point
# ====================================================================

def run_single_point(
    K: int,
    q: float,
    alpha: float,
    args: dict,
    save_example_ts: int = 0,
) -> dict:
    """Run R realizations for a single (K,q,alpha). Uses cache if valid.

    Returns summary dict for this point.
    """
    R = args["R"]
    base_seed = args["base_seed"]
    theta_base = args["theta_base"]
    theta_max = alpha * theta_base  # effective threshold
    meta = _build_meta(args, alpha)

    cache_path = CACHE_DIR / _cache_filename(K, q, alpha, R, base_seed)
    cached = _load_cache(cache_path, meta)

    if cached is not None:
        # Cache hit
        T_rec = cached["T_rec"]
        A_res = cached["A_res"]
        E_lost_max = cached["E_lost_max"]
        S_min = cached["S_min"]
        t_S_min = cached["t_S_min"]
        unrec = cached["unrecovered"].astype(bool)
        seeds = cached["seeds"]
        print(f"    [cache hit] {cache_path.name}")
    else:
        # Compute
        seeds = np.array([base_seed + r for r in range(R)])
        T_rec = np.full(R, np.nan)
        A_res = np.zeros(R)
        E_lost_max = np.zeros(R, dtype=int)
        S_min = np.zeros(R)
        t_S_min = np.zeros(R)
        unrec = np.zeros(R, dtype=bool)

        example_ts_data: dict = {}

        for r in range(R):
            seed = int(seeds[r])
            t0 = time.time()

            metrics, timeseries = simulate_cascade_recovery_ws(
                K=K, q=q, seed=seed,
                t_max=args["t_max"],
                dt=args["dt"],
                t_shock=args["t_shock"],
                shock_mode=args["shock_mode"],
                kappa=args["kappa"],
                theta_max=theta_max,
                fail_duration=args["fail_duration"],
                check_dt=args["check_dt"],
                repair_mean=args["repair_mean"],
                repair_dist=args["repair_dist"],
                retry_delay=args["retry_delay"],
                max_retries=args["max_retries"],
                eps=args["eps"],
                hold_time=args["hold_time"],
                N=args["N"],
                gen_ratio=args["gen_ratio"],
                Pmax=args["Pmax"],
            )

            T_rec[r] = metrics["T_rec"]
            A_res[r] = metrics["A_res"]
            E_lost_max[r] = metrics["E_lost_max"]
            S_min[r] = metrics["S_min"]
            t_S_min[r] = metrics["t_S_min"]
            unrec[r] = metrics["unrecovered"]

            # Optionally save example timeseries
            if r < save_example_ts:
                example_ts_data[f"ex{r}_t"] = timeseries["t_array"]
                example_ts_data[f"ex{r}_S"] = timeseries["S_array"]
                example_ts_data[f"ex{r}_nfail"] = timeseries["n_failed_array"]

            elapsed = time.time() - t0
            status = ("UNREC" if metrics["unrecovered"]
                       else f"T={metrics['T_rec']:.1f}")
            print(
                f"      [{r+1:3d}/{R}] seed={seed}  {status:>8s}  "
                f"S_min={metrics['S_min']:.3f}  E={metrics['E_lost_max']:2d}  "
                f"({elapsed:.1f}s)"
            )

        _save_cache(
            cache_path, meta, seeds,
            T_rec, A_res, E_lost_max, S_min, t_S_min, unrec,
            example_ts=example_ts_data if example_ts_data else None,
        )
        print(f"    [saved] {cache_path.name}")

    # Build summary
    recovered_mask = ~unrec
    n_unrec = int(unrec.sum())

    summary = {
        "K": K,
        "q": round(q, 3),
        "alpha": round(alpha, 3),
        "R": R,
        "n_unrecovered": n_unrec,
        "frac_unrecovered": n_unrec / R,
    }

    if recovered_mask.any():
        T_rec_valid = T_rec[recovered_mask]
        summary["T_rec_median"] = float(np.median(T_rec_valid))
        summary["T_rec_mean"] = float(np.mean(T_rec_valid))
        summary["T_rec_q25"] = float(np.percentile(T_rec_valid, 25))
        summary["T_rec_q75"] = float(np.percentile(T_rec_valid, 75))
    else:
        summary["T_rec_median"] = None
        summary["T_rec_mean"] = None
        summary["T_rec_q25"] = None
        summary["T_rec_q75"] = None

    summary["A_res_median"] = float(np.median(A_res))
    summary["A_res_mean"] = float(np.mean(A_res))
    summary["S_min_median"] = float(np.median(S_min))
    summary["E_lost_max_median"] = float(np.median(E_lost_max))

    return summary


# ====================================================================
# 4. Print summary for one (K, q, alpha) point
# ====================================================================

def _print_point_summary(s: dict) -> None:
    """Print a one-line summary for a (K, q, alpha) point."""
    T_str = (f"{s['T_rec_median']:.2f}s" if s["T_rec_median"] is not None
             else "N/A")
    print(
        f"    K={s['K']:2d}  q={s['q']:.3f}  a={s['alpha']:.3f}  |  "
        f"T_rec_med={T_str:>8s}  "
        f"unrec={s['n_unrecovered']}/{s['R']}  "
        f"A_res_med={s['A_res_median']:.3f}  "
        f"S_min_med={s['S_min_median']:.3f}  "
        f"E_max_med={s['E_lost_max_median']:.0f}"
    )


# ====================================================================
# 5. Main scan loop (3D: K × q × alpha)
# ====================================================================

def scan(
    K_list: list[int],
    q_array: np.ndarray,
    alpha_array: np.ndarray,
    args: dict,
    save_example_ts: int = 0,
) -> list[dict]:
    """Scan all (K, q, alpha) combinations and return list of summaries."""
    total_points = len(K_list) * len(q_array) * len(alpha_array)
    print(f"\n{'='*70}")
    print(f"  WS Cascade Recovery Scan (3D)")
    print(f"  K_list={K_list}  |  {len(q_array)} q-points  "
          f"|  {len(alpha_array)} alpha-points  |  R={args['R']}")
    print(f"  theta_base={args['theta_base']:.3f}  |  "
          f"alpha range=[{alpha_array[0]:.3f}, {alpha_array[-1]:.3f}]")
    print(f"  Total grid points: {total_points}")
    print(f"{'='*70}\n")

    all_summaries: list[dict] = []
    wall_start = time.time()
    done = 0

    for K in K_list:
        print(f"  ── K = {K} ──")
        for qi, q_val in enumerate(q_array):
            q_val = round(float(q_val), 3)
            for ai, a_val in enumerate(alpha_array):
                a_val = round(float(a_val), 3)
                print(f"    q={q_val:.3f}  alpha={a_val:.3f}  "
                      f"(theta_max={a_val * args['theta_base']:.3f})  "
                      f"({done+1}/{total_points})")
                summary = run_single_point(
                    K, q_val, a_val, args, save_example_ts
                )
                _print_point_summary(summary)
                all_summaries.append(summary)
                done += 1
        print()

    wall_total = time.time() - wall_start

    # Save summary JSON
    summary_path = CACHE_DIR / "summary_ws_scan.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"  Summary saved -> {summary_path}")
    print(f"  Total wall time: {wall_total:.1f}s\n")

    return all_summaries


# ====================================================================
# 6. CLI
# ====================================================================

def cli():
    parser = argparse.ArgumentParser(
        description="Batch cascade recovery-time scanner on WS networks "
                    "with alpha sweep (theta_max = alpha * theta_base).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Grid specification
    parser.add_argument(
        "--K_list", type=str, default="4,6,8",
        help="Comma-separated K values (e.g. '4,6,8,10,12')")
    parser.add_argument(
        "--q_points", type=int, default=None,
        help="Number of evenly-spaced q values in [0,1] (e.g. 21)")
    parser.add_argument(
        "--q_list", type=str, default=None,
        help="Explicit comma-separated q values (overrides --q_points)")
    parser.add_argument("--R", type=int, default=20,
                        help="Repetitions per (K,q,alpha) point")
    parser.add_argument("--base_seed", type=int, default=2026)

    # Alpha sweep
    parser.add_argument(
        "--theta_base", type=float, default=1.0,
        help="Base angle threshold (radians). theta_max = alpha * theta_base")
    parser.add_argument(
        "--alpha_points", type=int, default=None,
        help="Number of evenly-spaced alpha values (e.g. 21)")
    parser.add_argument(
        "--alpha_min", type=float, default=0.5,
        help="Minimum alpha value (default 0.5)")
    parser.add_argument(
        "--alpha_max", type=float, default=2.5,
        help="Maximum alpha value (default 2.5)")
    parser.add_argument(
        "--alpha_list", type=str, default=None,
        help="Explicit comma-separated alpha values (overrides --alpha_points)")

    # Simulation parameters
    parser.add_argument("--t_max", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--t_shock", type=float, default=5.0)
    parser.add_argument("--shock_mode", type=str, default="betweenness",
                        choices=["betweenness", "random", "max_load_edge"])
    parser.add_argument("--kappa", type=float, default=5.0)
    parser.add_argument("--fail_duration", type=float, default=0.5)
    parser.add_argument("--check_dt", type=float, default=0.25)
    parser.add_argument("--repair_mean", type=float, default=5.0)
    parser.add_argument("--repair_dist", type=str, default="fixed",
                        choices=["fixed", "exponential"])
    parser.add_argument("--retry_delay", type=float, default=2.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--hold_time", type=float, default=2.0)
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--gen_ratio", type=float, default=0.5)
    parser.add_argument("--Pmax", type=float, default=1.0)

    # Output control
    parser.add_argument("--save_example_ts", type=int, default=0,
                        help="Save timeseries for first N realizations (0=none)")

    args = parser.parse_args()

    # Parse K_list
    K_list = [int(k.strip()) for k in args.K_list.split(",")]

    # Parse q grid
    if args.q_list is not None:
        q_array = np.array([round(float(x.strip()), 3)
                            for x in args.q_list.split(",")])
    elif args.q_points is not None:
        q_array = np.round(np.linspace(0, 1, args.q_points), 3)
    else:
        q_array = np.round(np.linspace(0, 1, 21), 3)

    # Parse alpha grid
    if args.alpha_list is not None:
        alpha_array = np.array([round(float(x.strip()), 3)
                                for x in args.alpha_list.split(",")])
    elif args.alpha_points is not None:
        alpha_array = np.round(
            np.linspace(args.alpha_min, args.alpha_max, args.alpha_points), 3
        )
    else:
        alpha_array = np.round(
            np.linspace(args.alpha_min, args.alpha_max, 21), 3
        )

    # Build args dict (excluding grid params and save_example_ts)
    exclude_keys = {
        "K_list", "q_points", "q_list",
        "alpha_points", "alpha_min", "alpha_max", "alpha_list",
        "save_example_ts",
    }
    sim_args = {k: v for k, v in vars(args).items()
                if k not in exclude_keys}

    scan(K_list, q_array, alpha_array, sim_args,
         save_example_ts=args.save_example_ts)


if __name__ == "__main__":
    cli()
