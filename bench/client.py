"""Streaming load generator for an OpenAI-compatible completions endpoint.

Two arrival modes:
  closed loop -- `concurrency` workers, each sends the next request as soon as
                 its previous one finishes (measures capacity at fixed load);
  open loop   -- Poisson arrivals at `rate` req/s regardless of completions
                 (exposes queueing; the mode a real service sees).

Every request yields one record; timing is client-side wall clock.
"""

import asyncio
import json
import random
import time

import aiohttp

from bench.workloads import Request


def _pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


async def _send(session, url, model, req: Request, t_base, extra_body, priority):
    body = {
        "model": model,
        "prompt": req.prompt,
        "max_tokens": req.output_len,
        "ignore_eos": True,  # vLLM extension: generate exactly max_tokens
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        # Token IDs in every chunk: a token can decode to empty text (a partial
        # UTF-8 character), so timing must follow tokens, not text.
        "return_token_ids": True,
        **extra_body,
    }
    if priority:  # vLLM --scheduling-policy priority: lower value runs first
        body["priority"] = len(req.prompt)
    rec = {"idx": req.idx, "cls": req.cls, "prefix_id": req.prefix_id,
           "input_len": len(req.prompt), "output_len": req.output_len}
    t0 = time.perf_counter()
    first = last = None
    gaps, n_out, err, done = [], 0, None, False
    try:
        async with session.post(url, json=body) as resp:
            if "X-Gateway-Wait" in resp.headers:
                rec["gateway_wait"] = float(resp.headers["X-Gateway-Wait"])
            if resp.status != 200:
                err = f"HTTP {resp.status}: {(await resp.text())[:200]}"
            else:
                async for line in resp.content:
                    if not line.startswith(b"data: "):
                        continue
                    data = line[6:].strip()
                    if data == b"[DONE]":
                        done = True
                        break
                    chunk = json.loads(data)
                    if "error" in chunk:  # vLLM reports failures inside the stream
                        err = f"stream error: {json.dumps(chunk['error'])[:200]}"
                        break
                    if chunk.get("usage"):
                        n_out = chunk["usage"]["completion_tokens"]
                    if chunk.get("choices") and chunk["choices"][0].get("token_ids"):
                        now = time.perf_counter()
                        if first is None:
                            first = now
                        else:
                            gaps.append(now - last)
                        last = now
        if err is None and not (done and n_out):
            err = "incomplete stream"
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        err = repr(e)
    t1 = time.perf_counter()

    rec["start"] = t0 - t_base
    rec["e2e"] = t1 - t0
    rec["error"] = err
    rec["n_out"] = n_out
    if first is not None and err is None:
        rec["ttft"] = first - t0
        rec["tpot"] = (last - first) / (n_out - 1) if n_out > 1 else None
        rec["itl_p50"] = _pct(gaps, 0.50)
        rec["itl_p99"] = _pct(gaps, 0.99)
        rec["itl_max"] = max(gaps, default=None)
    return rec


async def run(url, model, reqs: list[Request], *, concurrency=None, rate=None,
              seed=0, extra_body=None, priority=False, timeout_s=1800):
    """Send all requests and return their records in completion order.

    priority=True sets each request's vLLM priority to its prompt length
    (shortest first), for comparing against vLLM's priority scheduling.
    """
    assert (concurrency is None) != (rate is None), "set exactly one of concurrency/rate"
    extra_body = extra_body or {}
    conn = aiohttp.TCPConnector(limit=0)  # client must never be the bottleneck
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    records = []
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        t_base = time.perf_counter()

        async def one(r):
            records.append(await _send(session, url, model, r, t_base, extra_body, priority))

        if concurrency:
            queue = list(reversed(reqs))

            async def worker():
                while queue:
                    await one(queue.pop())

            await asyncio.gather(*(worker() for _ in range(concurrency)))
        else:
            # Absolute schedule so event-loop jitter does not accumulate.
            rng = random.Random(seed)
            tasks, t_next = [], 0.0
            for r in reqs:
                await asyncio.sleep(max(0.0, t_base + t_next - time.perf_counter()))
                tasks.append(asyncio.create_task(one(r)))
                t_next += rng.expovariate(rate)
            await asyncio.gather(*tasks)
    return records
