from bench.workloads import build

W = {"classes": [
    {"name": "chat", "input": [100, 200], "output": 64, "weight": 3,
     "shared_prefix": 50, "num_prefixes": 2},
    {"name": "doc", "input": 1000, "output": [10, 20]},
]}


def test_build_is_deterministic_and_respects_spec():
    a, b = build(W, 200, seed=1), build(W, 200, seed=1)
    assert [r.prompt for r in a] == [r.prompt for r in b]
    chat = [r for r in a if r.cls == "chat"]
    assert all(100 <= len(r.prompt) <= 200 for r in chat)
    assert all(len(r.prompt) == 1000 and r.prefix_id is None for r in a if r.cls == "doc")
    same = [r for r in chat if r.prefix_id == chat[0].prefix_id]
    assert all(r.prompt[:50] == same[0].prompt[:50] for r in same)
    assert 0.6 < len(chat) / len(a) < 0.9
