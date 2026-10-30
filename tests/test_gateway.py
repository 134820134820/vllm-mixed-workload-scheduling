import asyncio

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.__main__ import make_app
from gateway.admission import Admission


def test_proxy_passes_through_and_rejects_bad_costs():
    async def main():
        async def completions(request):
            return web.json_response({"ok": True})

        upstream = web.Application()
        upstream.router.add_post("/v1/completions", completions)
        async with TestServer(upstream) as up:
            adm = Admission(budget=1024, aging=1000)
            app = make_app(str(up.make_url("")).rstrip("/"), adm,
                           max_model_len=512, block=16)
            async with TestClient(TestServer(app)) as gw:
                ok = await gw.post("/v1/completions", json={"prompt": [1, 2, 3], "max_tokens": 8})
                assert ok.status == 200 and await ok.json() == {"ok": True}
                assert "X-Gateway-Wait" in ok.headers
                for bad in ({"max_tokens": -1000}, {"max_tokens": 8, "n": -1},
                            {"max_tokens": "8"}):
                    r = await gw.post("/v1/completions", json={"prompt": [1, 2, 3], **bad})
                    assert r.status == 400, bad
                too_big = await gw.post("/v1/completions", json={"prompt": [1], "max_tokens": 4096})
                assert too_big.status == 429
            assert adm.in_flight == 0 and adm.num_waiting == 0

            # n samples are charged per sample: 2 x ceil((16 + 18) / 16) blocks
            # = 96 tokens, over a budget of 80 (pooling them would give 64).
            app = make_app(str(up.make_url("")).rstrip("/"), Admission(budget=80, aging=1000),
                           max_model_len=512, block=16)
            async with TestClient(TestServer(app)) as gw:
                r = await gw.post("/v1/completions",
                                  json={"prompt": list(range(16)), "max_tokens": 18, "n": 2})
                assert r.status == 429

    asyncio.run(main())
