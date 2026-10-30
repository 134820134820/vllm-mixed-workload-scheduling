"""Run an experiment file: every variant x workload x load point.

    python -m bench.run experiments/<name>.toml --model /path/to/Qwen3-8B

Layout: results/<experiment>/<variant>/<workload>/<load>/
          requests.jsonl  one record per request (client-side timing)
          metrics.jsonl.gz  vLLM gauges/counters + server CPU, every 0.5 s
          metrics_{before,after}.prom  raw scrapes, for histogram deltas
          gpu.csv         nvidia-smi samples
          meta.json       exact conditions of the case
        results/<experiment>/<variant>/{server,gateway}.log

A variant may set `args` (extra vllm flags), `gateway = [flags]` to put the
admission gateway in front of vLLM, and `priority = true` to send each
request's prompt length as its vLLM priority.

A case whose requests.jsonl already exists is skipped, so an interrupted
experiment resumes where it stopped. Delete the directory to re-run.
"""

import argparse
import asyncio
import json
import os
import subprocess
import time
import tomllib
from importlib.metadata import version

from bench import client, server, workloads
from bench.monitor import Monitor


def gpu_snapshot():
    q = lambda what: subprocess.run(
        ["nvidia-smi", f"--query-{what}", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    return {"gpu": q("gpu=name,memory.total,driver_version"),
            "compute_apps": q("compute-apps=pid,process_name,used_memory")}


def wait_gpu_idle(quiet_s=300, poll_s=15):
    """Block until no other process has used the GPU for `quiet_s` seconds.

    Measurements must not share the GPU, and a batch job that respawns workers
    leaves short gaps; grabbing the card in such a gap would starve its next
    worker, hence the quiet period rather than a single idle check.
    """
    quiet_since, announced = time.time(), False
    while time.time() - quiet_since < quiet_s:
        if apps := gpu_snapshot()["compute_apps"]:
            quiet_since = time.time()
            if not announced:
                print(f"waiting for GPU to be free; busy with:\n{apps}", flush=True)
                announced = True
        time.sleep(poll_s)


def load_tag(load):
    return f"c{load['concurrency']}" if "concurrency" in load else f"r{load['rate']:g}"


def run_case(base_url, entry_url, wl, load, seed, out_dir, pids, meta, priority):
    reqs = workloads.build(wl, load["requests"], seed=seed)
    server.reset_prefix_cache(base_url)
    meta = {**meta, "workload": wl, "load": load, "seed": seed,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"), **gpu_snapshot()}
    t0 = time.perf_counter()
    gateway_url = entry_url if entry_url != base_url else None
    with Monitor(base_url, out_dir, pids, gateway_url):
        records = asyncio.run(client.run(
            f"{entry_url}/v1/completions", server.SERVED_NAME, reqs,
            concurrency=load.get("concurrency"), rate=load.get("rate"), seed=seed,
            priority=priority))
    meta["wall_s"] = time.perf_counter() - t0
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    # Written last, atomically: its presence marks the case as complete.
    with open(f"{out_dir}/requests.jsonl.tmp", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in sorted(records, key=lambda r: r["idx"]))
    os.replace(f"{out_dir}/requests.jsonl.tmp", f"{out_dir}/requests.jsonl")
    errors = sum(r["error"] is not None for r in records)
    print(f"    done in {meta['wall_s']:.0f}s, errors={errors}", flush=True)


def warmup(base_url):
    wl = {"classes": [{"name": "warmup", "input": [64, 2048], "output": 32}]}
    asyncio.run(client.run(f"{base_url}/v1/completions", server.SERVED_NAME,
                           workloads.build(wl, 32, seed=12345), concurrency=8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--model", required=True, help="model directory or HF id")
    ap.add_argument("--results", default="results")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--only", nargs="*", help="variant names to run")
    ap.add_argument("--no-wait-idle", action="store_true",
                    help="start even if other processes are using the GPU")
    a = ap.parse_args()

    with open(a.experiment, "rb") as f:
        exp = tomllib.load(f)
    base_url = f"http://127.0.0.1:{a.port}"
    root = os.path.join(a.results, exp["name"])
    common = exp.get("server", {}).get("args", [])

    first = True
    for var in exp["variant"]:
        if a.only and var["name"] not in a.only:
            continue
        cases = [(wl, load) for wl in exp["workload"]
                 for load in wl.get("loads", exp.get("loads"))]
        todo = [(wl, load) for wl, load in cases if not os.path.exists(
            os.path.join(root, var["name"], wl["name"], load_tag(load), "requests.jsonl"))]
        if not todo:
            continue
        var_dir = os.path.join(root, var["name"])
        os.makedirs(var_dir, exist_ok=True)
        args = common + var.get("args", [])
        if not a.no_wait_idle:
            # Long quiet period only before taking the card the first time.
            wait_gpu_idle(quiet_s=300 if first else 30)
            first = False
        print(f"[{var['name']}] starting vllm {' '.join(args)}", flush=True)
        proc = server.start(a.model, args, a.port, os.path.join(var_dir, "server.log"))
        procs, entry_url = [proc], base_url
        try:
            server.wait_ready(proc, base_url)
            warmup(base_url)
            if "gateway" in var:
                # The gateway sizes its budget from the engine's KV cache.
                entry_url = f"http://127.0.0.1:{a.port + 100}"
                gw = server.start_gateway(base_url, var["gateway"], a.port + 100,
                                          os.path.join(var_dir, "gateway.log"))
                procs.append(gw)
                server.wait_ready(gw, entry_url, timeout_s=60, path="/stats")
            meta = {"experiment": exp["name"], "variant": var["name"],
                    "server_args": args, "gateway_args": var.get("gateway"),
                    "priority": var.get("priority", False),
                    "model": os.path.basename(a.model.rstrip("/")),
                    "vllm": version("vllm"), "torch": version("torch")}
            for wl, load in todo:
                out_dir = os.path.join(var_dir, wl["name"], load_tag(load))
                os.makedirs(out_dir, exist_ok=True)
                print(f"  {wl['name']} {load_tag(load)} n={load['requests']}", flush=True)
                run_case(base_url, entry_url, wl, load, exp.get("seed", 0), out_dir,
                         [p.pid for p in procs], meta, var.get("priority", False))
        finally:
            for p in reversed(procs):
                server.stop(p)


if __name__ == "__main__":
    main()
