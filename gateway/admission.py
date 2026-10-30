"""KV-budget admission with shortest-cost-first ordering and aging.

A request's cost is the KV cache it can occupy: prompt + max output tokens.
Requests are released to the engine only while the costs in flight fit the
budget. With the budget at most the engine's KV capacity (the default), no
request reaches the engine without KV room reserved for its whole lifetime: the engine's queue holds no request that is waiting for KV,
and it never over-admits into preemption. (It can still queue briefly for its
per-step token budget.) Waiting requests are ordered by
    cost - aging * seconds_waited
so short requests go first, and a waiting request gains priority over time.
When the front request does not fit, nothing behind it is admitted (no
backfilling). A cheaper request that arrives later can still take the front
until the waiting one has aged past it: with aging 1000 tok/s, an 8k-token
request outranks a fresh 300-token one after about 7.7 s. Aging is not a hard
bound on waiting: for that, set max_wait.
"""
import asyncio
import itertools
import time


class Rejected(Exception):
    pass


class Admission:
    def __init__(self, budget: int, aging: float, max_wait: float | None = None):
        self.budget = budget      # tokens of KV cache the gateway may hand out
        self.aging = aging        # tokens of priority credit per second waited
        self.max_wait = max_wait  # reject after waiting this long (None: never)
        self.in_flight = 0
        self._waiting = {}        # seq -> (cost, t_arrival, future)
        self._seq = itertools.count()

    def _key(self, item, now):
        cost, t, _ = item
        return cost - self.aging * (now - t)

    def _dispatch(self):
        # Linear scan: simple, and cheap for thousands of waiters.
        now = time.monotonic()
        while self._waiting:
            seq = min(self._waiting, key=lambda s: (self._key(self._waiting[s], now), s))
            cost, _, fut = self._waiting[seq]
            if self.in_flight + cost > self.budget:
                return
            del self._waiting[seq]
            self.in_flight += cost
            fut.set_result(None)

    async def acquire(self, cost: int) -> float:
        """Wait for admission; returns seconds spent waiting."""
        if cost > self.budget:
            raise Rejected(f"request needs {cost} KV tokens, budget is {self.budget}")
        t0 = time.monotonic()
        seq = next(self._seq)
        fut = asyncio.get_running_loop().create_future()
        self._waiting[seq] = (cost, t0, fut)
        self._dispatch()
        try:
            await asyncio.wait_for(asyncio.shield(fut), self.max_wait)
        except BaseException as e:  # timeout, or the client went away
            if fut.done():          # admitted at the last moment: give it back
                self.release(cost)
            else:                   # it may have been the head blocking others
                del self._waiting[seq]
                self._dispatch()
            if isinstance(e, asyncio.TimeoutError):
                raise Rejected(f"waited more than {self.max_wait}s for admission") from None
            raise
        return time.monotonic() - t0

    def release(self, cost: int):
        self.in_flight -= cost
        self._dispatch()

    @property
    def num_waiting(self):
        return len(self._waiting)

