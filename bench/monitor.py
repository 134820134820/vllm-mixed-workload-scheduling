"""Side-channel observation while a case runs.

- vLLM Prometheus `/metrics`: every `vllm:` sample except histogram buckets
  and `*_created` timestamps is polled into a time series (running/waiting
  requests, KV cache usage, preemptions, prefix cache hits, ...). Full raw scrapes are also saved before
  and after the case so histogram deltas (e.g. queue time) can be computed.
- GPU: `nvidia-smi` in loop mode writes a CSV.
- Gateway (if any): tokens in flight and requests waiting, from its /stats.
- Host: 1-minute load average, and CPU% of the server process trees (the API
  server and engine core are single-threaded Python hot loops, so one of them
  pinned near 100% means the GPU is waiting on the CPU).
"""

import gzip
import json
import os
import subprocess
import threading
import time
import urllib.request

import psutil

GPU_FIELDS = "timestamp,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu"


def scrape(base_url) -> str:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=5) as r:
        return r.read().decode()


def parse(text: str) -> dict[str, float]:
    """Flatten `vllm:` samples except histogram buckets into {series: value}."""
    out = {}
    for line in text.splitlines():
        if not line.startswith("vllm:") or "_bucket{" in line or "_created" in line:
            continue
        series, _, value = line.rpartition(" ")
        # Drop the model_name/engine labels that are constant within a run.
        name, _, labels = series.partition("{")
        labels = ",".join(l for l in labels.rstrip("}").split(",")
                          if l and not l.startswith(("model_name=", "engine=")))
        out[f"{name}{{{labels}}}" if labels else name] = float(value)
    return out


class Monitor:
    def __init__(self, base_url, out_dir, pids=(), gateway_url=None, interval=0.5):
        self.base_url, self.out_dir, self.interval = base_url, out_dir, interval
        self.pids, self.gateway_url = pids, gateway_url
        self._stop = threading.Event()

    def __enter__(self):
        with open(f"{self.out_dir}/metrics_before.prom", "w") as f:
            f.write(scrape(self.base_url))
        self._gpu = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={GPU_FIELDS}", "--format=csv,nounits",
             f"-lms={int(self.interval * 1000)}"],
            stdout=open(f"{self.out_dir}/gpu.csv", "w"), stderr=subprocess.DEVNULL)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _procs(self):
        roots = [psutil.Process(pid) for pid in self.pids]
        return [p for r in roots for p in (r, *r.children(recursive=True))]

    def _loop(self):
        t0 = time.time()
        procs = self._procs()
        for p in procs:
            p.cpu_percent()  # first call only primes the counter
        with gzip.open(f"{self.out_dir}/metrics.jsonl.gz", "wt") as f:
            while not self._stop.wait(self.interval):
                try:  # a missed sample is fine; a dead sampler thread is not
                    sample = parse(scrape(self.base_url))
                    if self.gateway_url:
                        with urllib.request.urlopen(f"{self.gateway_url}/stats", timeout=5) as r:
                            gw = json.load(r)
                        sample["gw_in_flight"], sample["gw_waiting"] = gw["in_flight"], gw["waiting"]
                except (OSError, ValueError):
                    continue
                sample["t"] = round(time.time() - t0, 3)
                sample["loadavg1"] = os.getloadavg()[0]
                for p in procs:
                    try:
                        sample[f"cpu:{p.name()}:{p.pid}"] = p.cpu_percent()
                    except psutil.NoSuchProcess:
                        pass
                f.write(json.dumps(sample) + "\n")

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._gpu.terminate()
        self._gpu.wait()
        with open(f"{self.out_dir}/metrics_after.prom", "w") as f:
            f.write(scrape(self.base_url))

