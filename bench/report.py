"""Summarize an experiment's raw results into a table and charts.

    python -m bench.report results/<experiment>

Writes results/<experiment>/summary.csv (one row per case and request class)
and one PNG per workload and arrival mode (plus one per request class for mixed
workloads): small multiples of throughput and P95 latencies vs
load, one line per server variant.
"""

import csv
import glob
import gzip
import json
import os
import re
import statistics
import sys

PCTS = {"p50": 0.50, "p95": 0.95, "p99": 0.99}


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q  # linear interpolation between closest ranks
    lo = int(k)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (k - lo)


# ---- server-side metrics from raw Prometheus scrapes -----------------------

_SAMPLE = re.compile(r'^(vllm:[a-z_:]+)(?:\{(.*)\})? ([0-9.eE+-]+|NaN|[+-]Inf)$')


def read_prom(path):
    """{(name, frozenset(labels without model/engine)): value}"""
    out = {}
    for line in open(path):
        m = _SAMPLE.match(line.strip())
        if not m:
            continue
        name, labels, v = m.groups()
        labels = frozenset(l for l in re.findall(r'(\w+="[^"]*")', labels or "")
                           if not l.startswith(("model_name=", "engine=")))
        out[name, labels] = float(v)
    return out


def counter_delta(before, after, name):
    return sum(v - before.get(k, 0.0) for k, v in after.items() if k[0] == name)


def server_side(case_dir):
    b = read_prom(f"{case_dir}/metrics_before.prom")
    a = read_prom(f"{case_dir}/metrics_after.prom")
    queries = counter_delta(b, a, "vllm:prefix_cache_queries_total")
    hits = counter_delta(b, a, "vllm:prefix_cache_hits_total")
    # Mean from the histogram's sum/count is exact; its buckets (first edge
    # 0.3 s) are too coarse for quantiles of short waits.
    q_n = counter_delta(b, a, "vllm:request_queue_time_seconds_count")
    out = {
        # Can differ between server starts (see README), so keep it per case.
        "kv_cache_tokens": int(re.search(r'kv_cache_size_tokens="(\d+)"', open(
            f"{case_dir}/metrics_before.prom").read()).group(1)),
        "queue_mean": counter_delta(b, a, "vllm:request_queue_time_seconds_sum") / q_n
        if q_n else None,
        "prefix_hit_rate": hits / queries if queries else None,
        "preemptions": counter_delta(b, a, "vllm:num_preemptions_total"),
    }
    samples = [json.loads(l) for l in gzip.open(f"{case_dir}/metrics.jsonl.gz", "rt")]
    if samples:
        kv = [s.get("vllm:kv_cache_usage_perc", 0) for s in samples]
        out["kv_usage_mean"] = statistics.fmean(kv)
        out["kv_usage_max"] = max(kv)
        out["running_max"] = max(s.get("vllm:num_requests_running", 0) for s in samples)
        out["waiting_max"] = max(s.get("vllm:num_requests_waiting", 0) for s in samples)
        if "gw_waiting" in samples[0]:
            out["gw_waiting_max"] = max(s["gw_waiting"] for s in samples)
        cpu = {}
        for s in samples:
            for k, v in s.items():
                if k.startswith("cpu:"):
                    cpu.setdefault(k[4:].rsplit(":", 1)[0], []).append(v)
        # Busiest process, and its mean; a single Python loop caps at ~100%.
        if cpu:
            proc, vals = max(cpu.items(), key=lambda kv: statistics.fmean(kv[1]))
            out["cpu_busiest"] = proc
            out["cpu_busiest_mean"] = statistics.fmean(vals)
    gpu = list(csv.reader(open(f"{case_dir}/gpu.csv")))[1:]
    if gpu:
        out["gpu_util_mean"] = statistics.fmean(float(r[1]) for r in gpu)
        out["gpu_mem_max_mib"] = max(float(r[2]) for r in gpu)
    return out


# ---- client-side metrics ----------------------------------------------------

def client_side(recs, window=None):
    """`window` = (t0, t1) of the whole case, so every request class and
    every outcome is measured over the same span."""
    ok = [r for r in recs if r["error"] is None]
    rejected = sum((r["error"] or "").startswith("HTTP 429") for r in recs)
    out = {"n": len(recs), "rejected": rejected, "errors": len(recs) - len(ok) - rejected}
    if not ok:
        return out
    t0, t1 = window or (min(r["start"] for r in recs),
                        max(r["start"] + r["e2e"] for r in recs))
    out["duration_s"] = t1 - t0
    out["req_per_s"] = len(ok) / out["duration_s"]
    out["out_tok_per_s"] = sum(r["n_out"] for r in ok) / out["duration_s"]
    # gateway_wait is already inside ttft/e2e (client clock); shown separately.
    for metric in ("ttft", "tpot", "e2e", "itl_max", "gateway_wait"):
        for tag, q in PCTS.items():
            out[f"{metric}_{tag}"] = pct([r.get(metric) for r in ok], q)
    return out


def summarize(root):
    rows = []
    for path in sorted(glob.glob(f"{root}/*/*/*/requests.jsonl")):
        case_dir = os.path.dirname(path)
        meta = json.load(open(f"{case_dir}/meta.json"))
        recs = [json.loads(l) for l in open(path)]
        load = meta["load"]
        base = {"variant": meta["variant"], "started": meta["started"],
                "workload": meta["workload"]["name"],
                "mode": "concurrency" if "concurrency" in load else "rate",
                "load": load.get("concurrency", load.get("rate"))}
        srv = server_side(case_dir)
        # Same total work in every variant, so time-to-finish is the fair
        # throughput comparison; drain shows where any difference arises.
        srv["drain_s"] = max(r["start"] + r["e2e"] for r in recs) - max(r["start"] for r in recs)
        classes = sorted({r["cls"] for r in recs})
        window = (min(r["start"] for r in recs), max(r["start"] + r["e2e"] for r in recs))
        for cls in ["all"] + (classes if len(classes) > 1 else []):
            sub = recs if cls == "all" else [r for r in recs if r["cls"] == cls]
            rows.append({**base, "cls": cls, **client_side(sub, window),
                         **(srv if cls == "all" else {})})
    return rows


def write_csv(rows, path):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: f"{v:.6g}" if isinstance(v, float) else v for k, v in r.items()})


# ---- charts -----------------------------------------------------------------

# Series colors in fixed order, then background, text and grid colors.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"

PANELS = [("out_tok_per_s", "Output throughput", "tokens/s", 1),
          ("ttft_p95", "TTFT P95", "ms", 1e3),
          ("tpot_p95", "TPOT P95", "ms", 1e3),
          ("itl_max_p95", "Longest stall per request, P95", "ms", 1e3),
          ("e2e_p95", "End-to-end latency P95", "s", 1)]


def plot_panels(series, panels, path, title, mode):
    """Small multiples: one panel per (key, title, unit, scale[, cls]), one line
    per (label, rows) in `series`, x = load. A panel with `cls` only uses that
    request class's rows. Colors follow the order of `series`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels) + 0.4, 3.4),
                             facecolor=SURFACE, squeeze=False)
    for ax, (key, ptitle, unit, scale, *cls) in zip(axes[0], panels):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for i, (label, rows) in enumerate(series):
            pts = sorted((r["load"], r[key] * scale) for r in rows
                         if r.get(key) is not None and (not cls or r["cls"] == cls[0]))
            if pts:
                xs, ys = zip(*pts)
                # Earlier series on top, so a line that coincides stays visible.
                ax.plot(xs, ys, color=SERIES[i], linewidth=2, marker="o", markersize=6,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, label=label,
                        zorder=3 + len(series) - i)
        if mode == "concurrency":
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_title(ptitle, loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel("concurrent requests" if mode == "concurrency" else "arrival rate (req/s)")
        ax.set_ylabel(unit)
        ax.set_ylim(bottom=0)
    fig.suptitle(title, x=0.01, y=0.98, ha="left", fontsize=12, color=INK)
    if len(series) > 1:  # own row under the title, so the two never collide
        fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper left",
                   bbox_to_anchor=(0.005, 0.91), ncol=min(len(series), 6),
                   frameon=False, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.82 if len(series) > 1 else 0.9))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_workload(rows, workload, mode, variants, path, cls="all"):
    sel = [r for r in rows if (r["workload"], r["mode"], r["cls"]) == (workload, mode, cls)]
    series = [(v, [r for r in sel if r["variant"] == v]) for v in variants]
    suffix = "" if cls == "all" else f" — {cls} requests"
    plot_panels(series, PANELS, path, f"{workload}{suffix}", mode)


def main():
    root = sys.argv[1].rstrip("/")
    rows = summarize(root)
    write_csv(rows, f"{root}/summary.csv")
    # In run order, i.e. as listed in the experiment file, so colors are stable.
    variants = list(dict.fromkeys(r["variant"] for r in sorted(rows, key=lambda r: r["started"])))
    os.makedirs(f"{root}/plots", exist_ok=True)
    for wl, mode in dict.fromkeys((r["workload"], r["mode"]) for r in rows):
        for cls in dict.fromkeys(r["cls"] for r in rows if r["workload"] == wl):
            name = wl + ("" if mode == "concurrency" else "-open") + ("" if cls == "all" else f".{cls}")
            plot_workload(rows, wl, mode, variants, f"{root}/plots/{name}.png", cls)
    print(f"{len(rows)} rows -> {root}/summary.csv, plots in {root}/plots/")


if __name__ == "__main__":
    main()
