import asyncio

import pytest

from gateway.admission import Admission, Rejected


async def _run_order(adm, *, later_after=0.01):
    """"big" fills the budget; while it runs, a long request and then three
    shorts queue. Returns the admission order."""
    order = []

    async def req(name, cost, hold):
        await adm.acquire(cost)
        order.append(name)
        await asyncio.sleep(hold)
        adm.release(cost)

    tasks = [asyncio.create_task(req("big", 100, 0.05))]
    await asyncio.sleep(0.01)
    tasks.append(asyncio.create_task(req("long", 90, 0.01)))
    await asyncio.sleep(later_after)
    tasks += [asyncio.create_task(req(f"s{i}", 10, 0.01)) for i in range(3)]
    await asyncio.gather(*tasks)
    return order


def test_short_requests_go_first():
    order = asyncio.run(_run_order(Admission(budget=100, aging=10.0)))
    assert order == ["big", "s0", "s1", "s2", "long"]


def test_aged_request_is_not_backfilled_past():
    # With strong aging the long request is at the front by the time the
    # shorts arrive; it does not fit, and nothing behind it may slip past.
    adm = Admission(budget=100, aging=1e6)
    order = asyncio.run(_run_order(adm))
    assert order[:2] == ["big", "long"]
    assert adm.in_flight == 0 and adm.num_waiting == 0


def test_cheaper_newcomer_can_take_the_front():
    # At 1000 tok/s the long request earns only tens of tokens of credit in
    # these milliseconds, so the later shorts (cost 10) still rank ahead of it.
    order = asyncio.run(_run_order(Admission(budget=100, aging=1000.0)))
    assert order[1:4] == ["s0", "s1", "s2"]


def test_timeout_and_over_budget_are_rejected():
    async def main():
        adm = Admission(budget=10, aging=0, max_wait=0.02)
        await adm.acquire(10)
        with pytest.raises(Rejected):
            await adm.acquire(5)
        adm.release(10)
        assert adm.in_flight == 0 and adm.num_waiting == 0
        with pytest.raises(Rejected):
            await adm.acquire(11)

    asyncio.run(main())


def test_cancelled_head_wakes_the_next_waiter():
    async def main():
        adm = Admission(budget=100, aging=1e6)
        await adm.acquire(60)
        big = asyncio.create_task(adm.acquire(80))
        await asyncio.sleep(0.01)
        small = asyncio.create_task(adm.acquire(20))
        await asyncio.sleep(0.01)
        big.cancel()
        await asyncio.sleep(0.01)
        assert small.done() and adm.in_flight == 80 and adm.num_waiting == 0

    asyncio.run(main())
