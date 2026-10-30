"""Admission gateway in front of one vLLM server (OpenAI-compatible).

    python -m gateway --upstream http://127.0.0.1:8100 --port 8200

Handles POST /v1/completions and /v1/chat/completions: computes each request's
KV cost (prompt tokens + max output tokens), waits for admission (see
admission.py), then streams the upstream response through unchanged. The time
spent waiting is returned in the `X-Gateway-Wait` header; requests that wait
longer than --max-wait or can never fit get HTTP 429, and an upstream failure
before any response gets HTTP 502. GET /stats exposes the queue for
monitoring.

Known limit: costs are counted per request, so prompts sharing a cached
prefix are charged for it repeatedly and the budget is used conservatively.
"""

import argparse
import json
import re
import urllib.request

from aiohttp import ClientError, ClientSession, ClientTimeout, TCPConnector, web

from gateway.admission import Admission, Rejected

SESSION = web.AppKey("session", ClientSession)


def kv_cache_shape(upstream):
    """(KV cache capacity in tokens, block size) as reported by vLLM."""
    with urllib.request.urlopen(f"{upstream}/metrics", timeout=10) as r:
        text = r.read().decode()
    get = lambda label: int(re.search(rf'{label}="(\d+)"', text).group(1))
    return get("kv_cache_size_tokens"), get("block_size")


async def prompt_tokens(session, upstream, body):
    """Token count of the prompt; token-ID prompts need no round trip."""
    prompt = body.get("prompt")
    if isinstance(prompt, list) and prompt and all(isinstance(t, int) for t in prompt):
        return len(prompt)
    if isinstance(prompt, list):
        raise web.HTTPBadRequest(text="batched prompts are not supported by the gateway")
    key = "messages" if "messages" in body else "prompt"
    async with session.post(f"{upstream}/tokenize",
                            json={"model": body.get("model"), key: body.get(key)}) as r:
        if r.status != 200:
            error = web.HTTPBadRequest if r.status < 500 else web.HTTPBadGateway
            raise error(text=f"tokenize failed: {await r.text()}")
        return (await r.json())["count"]


def make_app(upstream, adm, max_model_len, block):
    async def on_startup(app):
        # No connection limit or total timeout: streams can be long. No
        # keep-alive either: reusing an idle upstream connection races with
        # the server closing it (seen as ServerDisconnectedError -> HTTP 500
        # under load), and a fresh localhost connect costs ~0.1 ms.
        app[SESSION] = ClientSession(connector=TCPConnector(limit=0, force_close=True),
                                       timeout=ClientTimeout(total=None))

    async def on_cleanup(app):
        await app[SESSION].close()

    def upstream_error(e, headers=None):
        print(f"gateway: upstream error {e!r}", flush=True)
        return web.json_response(
            {"error": {"message": f"upstream error: {e!r}", "type": "upstream"}},
            status=502, headers=headers)

    async def proxy(request):
        session = request.app[SESSION]
        body = await request.json()
        try:
            n_in = await prompt_tokens(session, upstream, body)
        except ClientError as e:
            return upstream_error(e)
        n_out = body.get("max_tokens") or body.get("max_completion_tokens") \
            or max_model_len - n_in
        n = body.get("n") or 1
        # The cost is the client's own claim; a non-positive one would hand
        # budget back and let everyone else over-admit.
        if not all(isinstance(v, int) and v > 0 for v in (n_out, n)):
            raise web.HTTPBadRequest(text="max_tokens and n must be positive integers")
        # KV is allocated in whole blocks, per sample. Charging the prompt to
        # every sample over-reserves when it is shared, but never under-reserves.
        cost = n * (-(-(n_in + n_out) // block) * block)
        try:
            wait = await adm.acquire(cost)
        except Rejected as e:
            return web.json_response({"error": {"message": str(e), "type": "rejected"}},
                                     status=429)
        resp = None
        try:
            async with session.post(f"{upstream}{request.path}", json=body) as up:
                resp = web.StreamResponse(status=up.status, headers={
                    "Content-Type": up.headers.get("Content-Type", "application/json"),
                    "X-Gateway-Wait": f"{wait:.6f}"})
                await resp.prepare(request)
                async for chunk in up.content.iter_any():
                    await resp.write(chunk)
                await resp.write_eof()
                return resp
        except ClientError as e:
            # Not retried: the upstream may already have run the request.
            if resp is None:  # nothing sent yet: say so to the client
                return upstream_error(e, {"X-Gateway-Wait": f"{wait:.6f}"})
            print(f"gateway: upstream error {e!r}", flush=True)
            raise  # mid-stream: dropping the connection is the only signal left
        finally:
            adm.release(cost)

    async def stats(request):
        return web.json_response({"budget": adm.budget, "in_flight": adm.in_flight,
                                  "waiting": adm.num_waiting})

    app = web.Application(client_max_size=64 * 2**20)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_post("/v1/completions", proxy)
    app.router.add_post("/v1/chat/completions", proxy)
    app.router.add_get("/stats", stats)
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--budget-frac", type=float, default=1.0,
                    help="fraction of the engine's KV cache the gateway hands out")
    ap.add_argument("--aging", type=float, default=1000.0,
                    help="priority credit, in tokens, per second waited")
    ap.add_argument("--max-wait", type=float, default=None,
                    help="reject a request after waiting this many seconds")
    a = ap.parse_args()

    with urllib.request.urlopen(f"{a.upstream}/v1/models", timeout=10) as r:
        max_model_len = json.load(r)["data"][0]["max_model_len"]
    capacity, block = kv_cache_shape(a.upstream)
    budget = int(capacity * a.budget_frac)
    adm = Admission(budget, a.aging, a.max_wait)
    print(f"gateway: budget {budget} KV tokens, aging {a.aging} tok/s, "
          f"max_wait {a.max_wait}", flush=True)
    # handler_cancellation: a client that disconnects while queued is dropped
    # from the queue instead of being admitted later for nobody.
    web.run_app(make_app(a.upstream, adm, max_model_len, block), host="127.0.0.1",
                port=a.port, access_log=None, handler_cancellation=True)


if __name__ == "__main__":
    main()
