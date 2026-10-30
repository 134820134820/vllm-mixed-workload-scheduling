"""Start / stop the processes of one server configuration: `vllm serve`, and
optionally the admission gateway in front of it."""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

SERVED_NAME = "model"


def _open_log(path, cmd):
    """Append, never overwrite: a resumed experiment restarts processes, and
    the earlier logs are the evidence for the cases already recorded."""
    log = open(path, "a")
    log.write(f"\n===== {time.strftime('%Y-%m-%dT%H:%M:%S')} {' '.join(cmd)}\n\n")
    log.flush()
    return log


def start(model_path, args, port, log_path) -> subprocess.Popen:
    # Localhost only: dev mode below also exposes admin endpoints.
    cmd = ["vllm", "serve", model_path, "--served-model-name", SERVED_NAME,
           "--host", "127.0.0.1", "--port", str(port), *args]
    # Dev mode exposes POST /reset_prefix_cache, used to isolate cases.
    env = {**os.environ, "VLLM_SERVER_DEV_MODE": "1"}
    # Use the vllm installed next to this interpreter, whatever PATH is.
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env["PATH"]
    log = _open_log(log_path, cmd)
    # Own process group so the engine-core child dies with it.
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)


def start_gateway(upstream, args, port, log_path) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "gateway", "--upstream", upstream, "--port", str(port), *args]
    log = _open_log(log_path, cmd)
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def wait_ready(proc, base_url, timeout_s=900, path="/health"):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{proc.args[:2]} exited with code {proc.returncode}, see its log")
        try:
            with urllib.request.urlopen(f"{base_url}{path}", timeout=2):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"{base_url} did not become ready")


def reset_prefix_cache(base_url, attempts=10):
    """vLLM answers 200 with {"success": false} while blocks are still held."""
    for _ in range(attempts):
        req = urllib.request.Request(f"{base_url}/reset_prefix_cache", method="POST")
        with urllib.request.urlopen(req, timeout=30) as r:
            if json.load(r).get("success"):
                return
        time.sleep(1)
    raise RuntimeError("prefix cache reset kept failing")


def _group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def stop(proc):
    """Stop the whole process group: children (e.g. the engine core) can
    outlive the parent and keep holding GPU memory."""
    pgid = proc.pid  # started with start_new_session, so pgid == pid
    if _group_alive(pgid):
        os.killpg(pgid, signal.SIGINT)
    deadline = time.time() + 60
    while _group_alive(pgid) and time.time() < deadline:
        proc.poll()  # reap the parent so it does not linger as a zombie
        time.sleep(0.5)
    if _group_alive(pgid):
        os.killpg(pgid, signal.SIGKILL)
    proc.wait()
