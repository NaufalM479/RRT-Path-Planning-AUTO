"""
runs.py  –  ablation study runner for ReRT
Cycles through every configuration in pref.CONFIGS, runs TRIALS_PER_CONFIG
independent trials each, and writes aggregated metrics to a CSV file.
One representative figure is saved per configuration.
"""

import csv
import os
import time

import numpy as np

from rert import ReRT
import pref


# ─── metric keys written to CSV ──────────────────────────────────────────────
_FIELDS = [
    "config_id",
    "epoch",
    "expand_dis",
    "trial",
    "success",
    "iterations_used",
    "nodes_explored",
    "path_length",
    "exec_time_s",
]


def _run_single(cfg: dict, seed: int):
    """Run one trial.  Returns a metric dict."""
    np.random.seed(seed)

    rrt = ReRT(
        start=pref.START,
        finish=pref.FINISH,
        obstacles=pref.OBSTACLES,
        bounds=pref.BOUNDS,
        expand_dis=cfg["expand_dis"],
        epoch=cfg["epoch"],
    )

    t0 = time.perf_counter()
    path, iterations = rrt.path_planning()
    elapsed = time.perf_counter() - t0

    return {
        "path":            path,
        "rrt":             rrt,
        "success":         int(path is not None),
        "iterations_used": iterations,
        "nodes_explored":  len(rrt.node_list),
        "path_length":     rrt.compute_path_length(path),
        "exec_time_s":     elapsed,
    }


def run_ablation():
    """
    Entry point.  Runs the full ablation study, prints progress, writes CSV,
    and saves one representative figure per configuration.
    Returns a list of all per-trial row dicts.
    """
    os.makedirs(pref.FIG_DIR, exist_ok=True)

    all_rows = []

    print(f"\n{'='*60}")
    print(f"  ReRT Ablation Study")
    print(f"  {len(pref.CONFIGS)} configs × {pref.TRIALS_PER_CONFIG} trials"
          f" = {len(pref.CONFIGS) * pref.TRIALS_PER_CONFIG} total runs")
    print(f"{'='*60}\n")

    with open(pref.CSV_OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()

        for cfg_idx, cfg in enumerate(pref.CONFIGS):
            config_id = f"cfg_{cfg_idx:03d}_e{cfg['epoch']}_d{cfg['expand_dis']}"
            print(f"[{cfg_idx+1:2d}/{len(pref.CONFIGS)}]  {config_id}")

            best_result = None   # track a successful trial for the figure

            for trial in range(pref.TRIALS_PER_CONFIG):
                seed = pref.BASE_SEED + trial
                result = _run_single(cfg, seed)

                row = {
                    "config_id":       config_id,
                    "epoch":           cfg["epoch"],
                    "expand_dis":      cfg["expand_dis"],
                    "trial":           trial,
                    "success":         result["success"],
                    "iterations_used": result["iterations_used"],
                    "nodes_explored":  result["nodes_explored"],
                    "path_length":     f"{result['path_length']:.4f}",
                    "exec_time_s":     f"{result['exec_time_s']:.6f}",
                }

                writer.writerow(row)
                all_rows.append(row)

                status = "✓" if result["success"] else "✗"
                print(
                    f"     trial {trial:2d}  {status}  "
                    f"iters={result['iterations_used']:>5}  "
                    f"nodes={result['nodes_explored']:>5}  "
                    f"len={result['path_length']:>7.2f}  "
                    f"t={result['exec_time_s']:.3f}s"
                )

                # keep the first successful result for the figure
                if best_result is None and result["success"]:
                    best_result = result
                # fallback: keep last result if none succeeded
                if trial == pref.TRIALS_PER_CONFIG - 1 and best_result is None:
                    best_result = result

            # save one figure per config
            fig_path = os.path.join(pref.FIG_DIR, f"{config_id}.png")
            best_result["rrt"].save_figure(best_result["path"], fig_path)
            print(f"     → figure saved: {fig_path}\n")

    _print_summary(all_rows)
    print(f"\nResults written to: {pref.CSV_OUTPUT}")
    return all_rows


# ─── summary table ────────────────────────────────────────────────────────────

def _print_summary(rows):
    from collections import defaultdict

    agg = defaultdict(lambda: {
        "success": [], "iter": [], "nodes": [], "length": [], "time": []
    })

    for r in rows:
        key = (r["epoch"], r["expand_dis"])
        agg[key]["success"].append(int(r["success"]))
        agg[key]["iter"].append(int(r["iterations_used"]))
        agg[key]["nodes"].append(int(r["nodes_explored"]))
        agg[key]["length"].append(float(r["path_length"]))
        agg[key]["time"].append(float(r["exec_time_s"]))

    print(f"\n{'='*100}")
    print(f"  ABLATION EVALUATION METRICS  ({pref.TRIALS_PER_CONFIG} trials per config)")
    print(f"{'='*100}")
    print(f"{'epoch':>6} {'expand':>7} {'SR%':>6} {'fail':>5} "
          f"{'iter_min':>9} {'iter_max':>9} {'iter_avg':>9} "
          f"{'node_min':>9} {'node_max':>9} {'node_avg':>9} "
          f"{'len_min':>9} {'len_max':>9} {'len_avg':>9} "
          f"{'t_min':>8} {'t_max':>8} {'t_avg':>8}")
    print("-" * 100)

    for (epoch, expand_dis) in sorted(agg.keys()):
        d = agg[(epoch, expand_dis)]
        n = len(d["success"])

        sr           = 100 * np.mean(d["success"])
        fail_count   = n - sum(d["success"])

        iter_vals    = d["iter"]
        node_vals    = d["nodes"]
        time_vals    = d["time"]
        succ_lengths = [l for l, s in zip(d["length"], d["success"]) if s and not np.isnan(l)]

        len_min = f"{min(succ_lengths):.1f}"  if succ_lengths else "N/A"
        len_max = f"{max(succ_lengths):.1f}"  if succ_lengths else "N/A"
        len_avg = f"{np.mean(succ_lengths):.1f}" if succ_lengths else "N/A"

        print(
            f"{epoch:>6} {expand_dis:>7} {sr:>5.0f}% {fail_count:>5} "
            f"{min(iter_vals):>9} {max(iter_vals):>9} {np.mean(iter_vals):>9.1f} "
            f"{min(node_vals):>9} {max(node_vals):>9} {np.mean(node_vals):>9.1f} "
            f"{len_min:>9} {len_max:>9} {len_avg:>9} "
            f"{min(time_vals):>8.3f} {max(time_vals):>8.3f} {np.mean(time_vals):>8.3f}"
        )

    print("=" * 100)
