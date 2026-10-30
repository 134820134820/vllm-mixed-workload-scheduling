"""Deterministic synthetic workloads.

Prompts are random token IDs sent through the completions API, so input and
output lengths are exact and independent of any tokenizer or text corpus.
Shared prefixes are literal token sequences reused across requests, which is
what vLLM's automatic prefix caching can hit on.
"""

import random
from dataclasses import dataclass

# Qwen3 regular vocabulary is [0, 151643); stay well inside it and away from
# the low IDs to avoid control/whitespace tokens.
TOKEN_LO, TOKEN_HI = 1000, 150000


@dataclass
class Request:
    idx: int
    cls: str
    prompt: list[int]
    output_len: int
    prefix_id: int | None  # which shared prefix it uses, None if unique


def _length(spec, rng):
    """A length is either an int or an inclusive [lo, hi] uniform range."""
    if isinstance(spec, int):
        return spec
    lo, hi = spec
    return rng.randint(lo, hi)


def build(workload: dict, num_requests: int, seed: int = 0) -> list[Request]:
    """Build requests from a workload spec.

    workload = {"classes": [{"name", "input", "output", "weight"?,
                             "shared_prefix"?, "num_prefixes"?}, ...]}
    `input` counts the whole prompt, shared prefix included.
    """
    rng = random.Random(seed)
    classes = workload["classes"]
    weights = [c.get("weight", 1.0) for c in classes]

    prefixes = {}
    for ci, c in enumerate(classes):
        n = c.get("shared_prefix", 0)
        for p in range(c.get("num_prefixes", 1) if n else 0):
            prefixes[ci, p] = [rng.randrange(TOKEN_LO, TOKEN_HI) for _ in range(n)]

    reqs = []
    for i in range(num_requests):
        ci = rng.choices(range(len(classes)), weights)[0]
        c = classes[ci]
        n_in = _length(c["input"], rng)
        prefix_id, prompt = None, []
        if c.get("shared_prefix", 0):
            prefix_id = rng.randrange(c.get("num_prefixes", 1))
            prompt = list(prefixes[ci, prefix_id])
        assert n_in > len(prompt), f"class {c['name']}: input must exceed shared_prefix"
        prompt += [rng.randrange(TOKEN_LO, TOKEN_HI) for _ in range(n_in - len(prompt))]
        reqs.append(Request(i, c["name"], prompt, _length(c["output"], rng), prefix_id))
    return reqs

