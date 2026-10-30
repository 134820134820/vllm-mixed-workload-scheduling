"""Figures for the README, from the raw results.

    python scripts/figures.py   ->  docs/figures/*.png
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench.report import plot_panels, summarize  # noqa: E402

OUT = "docs/figures"


def pick(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def main():
    os.makedirs(OUT, exist_ok=True)

    # 1. Head-of-line blocking: in the mixed workload short requests wait as
    #    long as the long ones they queue behind.
    e1 = summarize("results/e1_baseline")
    plot_panels([
        ("short, in mixed", pick(e1, workload="mixed", cls="short")),
        ("long, in mixed", pick(e1, workload="mixed", cls="long_input")),
        ("short, alone", pick(e1, workload="short", cls="all")),
    ], [("ttft_p50", "TTFT P50", "s", 1), ("ttft_p95", "TTFT P95", "s", 1)],
        f"{OUT}/hol_blocking.png", "Short requests wait behind long ones (e1, closed loop)",
        "concurrency")

    # 2. Prefill chunk budget: a bigger budget stalls every running decode.
    e3 = summarize("results/e3_prefill_budget")
    variants = ["mbt512", "mbt2048", "mbt8192", "mbt16384"]
    plot_panels([(v.replace("mbt", "budget "), pick(e3, variant=v, cls="short"))
                 for v in variants],
                [("itl_max_p95", "Longest stall per request, P95", "ms", 1e3),
                 ("ttft_p95", "TTFT P95", "s", 1)],
                f"{OUT}/prefill_budget.png",
                "Per-step token budget, short requests in the mixed workload (e3)", "rate")

    # 3. Admission control: framework knobs vs the gateway.
    e4 = summarize("results/e4_admission")
    variants = ["fcfs", "lpt512", "vllm_priority", "gw_sjf", "gw_sjf_lpt512"]
    plot_panels([(v, pick(e4, variant=v)) for v in variants],
                [("ttft_p95", "Short requests: TTFT P95", "s", 1, "short"),
                 ("ttft_p95", "Long requests: TTFT P95", "s", 1, "long_input"),
                 ("preemptions", "Preemptions", "count", 1, "all")],
                f"{OUT}/admission.png", "Mixed workload, open loop (e4)", "rate")
    print(f"figures in {OUT}/")


if __name__ == "__main__":
    main()
