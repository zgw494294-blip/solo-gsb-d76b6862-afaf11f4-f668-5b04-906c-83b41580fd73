"""/v1/interval-risk 的功能与 422 路径测试。

参考实现枚举全部 2^n 事件赋值求给定端点组合下的顶事件概率，再枚举全部
2^k 个区间端点组合（k≤8）取最小/最大，与 ROBDD 稀疏多线性多项式的
结果逐项对照。
"""

import itertools
import random
from fractions import Fraction

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def f2s(f):
    return f"{f.numerator}/{f.denominator}"


def brute_force_extrema(events, gates, top, intervals):
    """返回 (min_p, max_p, min_endpoints, max_endpoints)，端点均为 Fraction。

    events: dict id -> 'n/d'；intervals: dict id -> (low_str, high_str)。
    并列按事件 id 字典序、low 优先（取数值最小的端点掩码）。
    """
    ev_ids = sorted(events)
    base_p = {eid: Fraction(events[eid]) for eid in ev_ids}
    lows = {eid: Fraction(intervals[eid][0]) for eid in intervals}
    highs = {eid: Fraction(intervals[eid][1]) for eid in intervals}
    interval_ids = sorted(intervals)
    gate = {g["id"]: (g["op"], g["inputs"]) for g in gates}

    def evaluate(state, node):
        if node in base_p:
            return state[node]
        op, inputs = gate[node]
        vals = [evaluate(state, i) for i in inputs]
        if op == "AND":
            return int(all(vals))
        if op == "OR":
            return int(any(vals))
        return int(not vals[0])

    def top_prob(endpoint_values):
        probs = dict(base_p)
        probs.update(endpoint_values)
        total = Fraction(0)
        for bits in itertools.product((0, 1), repeat=len(ev_ids)):
            state = dict(zip(ev_ids, bits))
            if evaluate(state, top):
                w = Fraction(1)
                for eid in ev_ids:
                    w *= probs[eid] if state[eid] else (Fraction(1) - probs[eid])
                total += w
        return total

    min_p = max_p = None
    min_ep = max_ep = None
    for mask in range(1 << len(interval_ids)):
        ep = {
            eid: (highs[eid] if (mask >> j) & 1 else lows[eid])
            for j, eid in enumerate(interval_ids)
        }
        value = top_prob(ep)
        if min_p is None:
            min_p = max_p = value
            min_ep = max_ep = dict(ep)
        else:
            if value < min_p:
                min_p, min_ep = value, dict(ep)
            if value > max_p:
                max_p, max_ep = value, dict(ep)
    return min_p, max_p, min_ep, max_ep


def make_body(ev_specs, gate_specs, top, intervals, limit):
    events = [{"id": k, "p": v} for k, v in ev_specs.items()]
    gates = [{"id": i, "op": op, "inputs": ins} for i, op, ins in gate_specs]
    return {
        "events": events,
        "gates": gates,
        "top": top,
        "intervals": {
            eid: {"low": lo, "high": hi} for eid, (lo, hi) in intervals.items()
        },
        "limit": limit,
    }


def post(body):
    return client.post("/v1/interval-risk", json=body)


# ---------- 基本功能 ----------

def test_or_tree_example():
    body = make_body(
        {"motor_overheat": "1/100", "coolant_loss": "2/100",
         "pump_fail": "1/200"},
        [("cooling_fail", "OR", ["coolant_loss", "pump_fail"]),
         ("top", "OR", ["motor_overheat", "cooling_fail"])],
        "top",
        {"coolant_loss": ("1/100", "3/100"),
         "pump_fail": ("0/1", "1/50")},
        "1/20",
    )
    r = post(body)
    assert r.status_code == 200, r.text
    data = r.json()
    # 单调 OR：最小取全 low，最大取全 high
    # min = 1/100 + (1/100 + 0 − 0)·99/100 = 199/10000
    # max = 1/100 + (3/100 + 1/50 − 3/5000)·99/100 = 29453/500000
    assert data["min_probability"] == "199/10000"
    assert data["max_probability"] == "29453/500000"
    assert data["min_endpoints"] == {
        "coolant_loss": "1/100", "pump_fail": "0/1"}
    assert data["max_endpoints"] == {
        "coolant_loss": "3/100", "pump_fail": "1/50"}
    # min=0.0199 < 0.05 < max≈0.0591 => uncertain
    assert Fraction(data["min_probability"]) < Fraction(body["limit"])
    assert Fraction(data["max_probability"]) > Fraction(body["limit"])
    assert data["verdict"] == "uncertain"


def test_verdicts():
    # safe：max <= limit
    body = make_body(
        {"a": "1/2", "b": "1/2"}, [("g", "OR", ["a", "b"])], "g",
        {"a": ("1/4", "1/2"), "b": ("1/4", "1/2")}, "3/4")
    r = post(body)
    assert r.status_code == 200, r.text
    assert r.json()["max_probability"] == "3/4"
    assert r.json()["verdict"] == "safe"

    # unsafe：min > limit
    body["limit"] = "1/4"
    r = post(body)
    assert r.status_code == 200, r.text
    assert r.json()["min_probability"] == "7/16"
    assert r.json()["verdict"] == "unsafe"


def test_single_interval_event_as_top():
    body = make_body(
        {"a": "1/2"}, [], "a", {"a": ("1/3", "2/3")}, "1/2")
    r = post(body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["min_probability"] == "1/3"
    assert data["max_probability"] == "2/3"
    assert data["min_endpoints"] == {"a": "1/3"}
    assert data["max_endpoints"] == {"a": "2/3"}
    assert data["verdict"] == "uncertain"


def test_constant_tree_endpoints():
    # 顶事件恒真（a OR NOT a）：概率恒为 1，端点任意，并列时 low 优先
    body = make_body(
        {"a": "1/2", "b": "1/2"},
        [("n", "NOT", ["a"]), ("g", "OR", ["a", "n"])], "g",
        {"a": ("0/1", "1/1"), "b": ("1/4", "3/4")}, "1/1")
    r = post(body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["min_probability"] == "1/1"
    assert data["max_probability"] == "1/1"
    assert data["min_endpoints"] == {"a": "0/1", "b": "1/4"}
    assert data["max_endpoints"] == {"a": "0/1", "b": "1/4"}
    assert data["verdict"] == "safe"

    # 顶事件恒假（a AND NOT a），min=0 > 1/2 不成立 => 不是 unsafe；
    # max=0 <= 0/1 为 safe
    body = make_body(
        {"a": "1/2"},
        [("n", "NOT", ["a"]), ("g", "AND", ["a", "n"])], "g",
        {"a": ("1/3", "2/3")}, "0/1")
    r = post(body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["min_probability"] == "0/1"
    assert data["max_probability"] == "0/1"
    assert data["verdict"] == "safe"


def test_unreferenced_interval_event():
    # 区间事件 b 未被顶事件引用：概率恒为 P(a)，b 的端点并列取 low
    body = make_body(
        {"a": "1/2", "b": "1/2"}, [("g", "NOT", ["a"])], "g",
        {"a": ("1/4", "3/4"), "b": ("1/4", "3/4")}, "1/2")
    r = post(body)
    assert r.status_code == 200, r.text
    data = r.json()
    # NOT a：P 随 a 增大而减小
    assert data["min_probability"] == "1/4"
    assert data["max_probability"] == "3/4"
    assert data["min_endpoints"] == {"a": "3/4", "b": "1/4"}
    assert data["max_endpoints"] == {"a": "1/4", "b": "1/4"}


def test_shared_reference():
    # (a AND b) OR (a AND NOT b) = a：共享 a 的两处引用必须同一状态
    body = make_body(
        {"a": "1/2", "b": "1/3"},
        [("g1", "AND", ["a", "b"]),
         ("n", "NOT", ["b"]),
         ("g2", "AND", ["a", "n"]),
         ("top", "OR", ["g1", "g2"])],
        "top",
        {"a": ("1/5", "4/5"), "b": ("0/1", "1/1")}, "1/2")
    r = post(body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["min_probability"] == "1/5"
    assert data["max_probability"] == "4/5"
    # b 不影响结果：并列时 b 一律取 low
    assert data["min_endpoints"] == {"a": "1/5", "b": "0/1"}
    assert data["max_endpoints"] == {"a": "4/5", "b": "0/1"}


def test_nonmonotone_interior_dependence():
    # XOR(a,b) = (a∨b)∧¬(a∧b)：极值仍在端点（多线性），但方向非单调
    body = make_body(
        {"a": "1/2", "b": "1/2"},
        [("o", "OR", ["a", "b"]),
         ("ad", "AND", ["a", "b"]),
         ("n", "NOT", ["ad"]),
         ("top", "AND", ["o", "n"])],
        "top",
        {"a": ("1/4", "3/4"), "b": ("1/4", "3/4")}, "1/2")
    r = post(body)
    assert r.status_code == 200, r.text
    data = r.json()
    # P = a+b-2ab：(low,low)=(1/4,1/4)->3/8，(high,high)=(3/4,3/4)->3/8，
    # (1/4,3/4) 或 (3/4,1/4) -> 1 − 2·(1/4)·(3/4) = 5/8
    assert data["min_probability"] == "3/8"
    assert data["max_probability"] == "5/8"
    # 最小并列掩码 0（全 low，字典序最小）
    assert data["min_endpoints"] == {"a": "1/4", "b": "1/4"}
    assert data["max_endpoints"] == {"a": "3/4", "b": "1/4"}


def test_tie_break_low_preferred():
    # 区间事件 b 不影响顶事件：并列时所有极值端点中 b 都取 low；
    # 同时验证按 id 字典序排列输出
    body = make_body(
        {"z": "1/2", "a": "1/2"}, [], "z",
        {"z": ("1/3", "2/3"), "a": ("1/3", "2/3")}, "1/2")
    r = post(body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert list(data["min_endpoints"]) == ["a", "z"]
    assert data["min_endpoints"] == {"a": "1/3", "z": "1/3"}
    assert data["max_endpoints"] == {"a": "1/3", "z": "2/3"}


def test_random_small_circuits():
    """随机 ≤8 事件、≤16 门、≤4 区间事件的故障树，逐例穷举端点对照。"""
    rng = random.Random(20260929)
    for _ in range(60):
        n = rng.randint(1, 8)
        ev_ids = [f"e{i}" for i in range(n)]
        ev_specs = {}
        for e in ev_ids:
            d = rng.randint(1, 6)
            ev_specs[e] = f"{rng.randint(0, d)}/{d}"
        gates = []
        available = list(ev_ids)
        for j in range(rng.randint(0, 16)):
            gid = f"g{j}"
            op = rng.choice(["AND", "OR", "NOT"]) if len(available) >= 2 else "NOT"
            if op == "NOT":
                inputs = [rng.choice(available)]
            else:
                k = rng.randint(2, min(8, len(available)))
                inputs = rng.sample(available, k)
            gates.append({"id": gid, "op": op, "inputs": inputs})
            available.append(gid)
        top = rng.choice(available)

        k = rng.randint(1, min(4, n))
        chosen = rng.sample(ev_ids, k)
        intervals = {}
        for eid in chosen:
            p = Fraction(ev_specs[eid])
            lo = Fraction(rng.randint(0, p.numerator * p.denominator),
                          p.denominator * p.denominator)
            hi = Fraction(rng.randint(p.numerator * p.denominator,
                                      p.denominator * p.denominator),
                          p.denominator * p.denominator)
            intervals[eid] = (lo, hi)

        body = make_body(
            ev_specs,
            [(g["id"], g["op"], g["inputs"]) for g in gates],
            top,
            {eid: (f2s(lo), f2s(hi)) for eid, (lo, hi) in intervals.items()},
            "1/2")
        r = post(body)
        assert r.status_code == 200, (r.text, ev_specs, gates, intervals)
        data = r.json()
        min_p, max_p, min_ep, max_ep = brute_force_extrema(
            ev_specs, gates, top, intervals)
        assert data["min_probability"] == f2s(min_p)
        assert data["max_probability"] == f2s(max_p)
        assert data["min_endpoints"] == {e: f2s(v) for e, v in min_ep.items()}
        assert data["max_endpoints"] == {e: f2s(v) for e, v in max_ep.items()}
        limit = Fraction(1, 2)
        expect = "safe" if max_p <= limit else ("unsafe" if min_p > limit
                                                else "uncertain")
        assert data["verdict"] == expect


# ---------- 422 错误路径 ----------

BASE = make_body(
    {"a": "1/2", "b": "1/3"},
    [("g", "OR", ["a", "b"])], "g",
    {"a": ("1/3", "2/3")}, "1/2")


def test_unknown_interval_event():
    body = make_body(
        {"a": "1/2"}, [], "a", {"x": ("0/1", "1/1")}, "1/2")
    r = post(body)
    assert r.status_code == 422
    assert "未声明" in r.json()["detail"]


def test_interval_outside_original_p():
    # low > 原 p
    body = make_body(
        {"a": "1/2"}, [], "a", {"a": ("3/4", "1/1")}, "1/2")
    r = post(body)
    assert r.status_code == 422
    assert "区间非法" in r.json()["detail"]

    # high < 原 p
    body = make_body(
        {"a": "1/2"}, [], "a", {"a": ("0/1", "1/4")}, "1/2")
    r = post(body)
    assert r.status_code == 422
    assert "区间非法" in r.json()["detail"]


def test_low_greater_than_high():
    body = make_body(
        {"a": "1/2"}, [], "a", {"a": ("3/4", "1/4")}, "1/2")
    r = post(body)
    assert r.status_code == 422
    assert "low" in r.json()["detail"] and "high" in r.json()["detail"]


def test_bad_interval_probability_format():
    body = make_body(
        {"a": "1/2"}, [], "a", {"a": ("x", "1/1")}, "1/2")
    r = post(body)
    assert r.status_code == 422
    assert "low" in r.json()["detail"] and "概率" in r.json()["detail"]


def test_bad_limit():
    body = dict(BASE)
    body["limit"] = "2/1"
    r = post(body)
    assert r.status_code == 422
    assert "limit" in r.json()["detail"]

    body["limit"] = "abc"
    r = post(body)
    assert r.status_code == 422
    assert "limit" in r.json()["detail"]


def test_intervals_count_limits():
    # 空 intervals
    body = dict(BASE)
    body["intervals"] = {}
    r = post(body)
    assert r.status_code == 422

    # 9 个区间
    evs = {f"e{i}": "1/2" for i in range(9)}
    intervals = {eid: ("1/4", "3/4") for eid in evs}
    body = make_body(evs, [], "e0", intervals, "1/2")
    r = post(body)
    assert r.status_code == 422


def test_extra_field_rejected():
    body = dict(BASE)
    body["intervals"]["a"] = {"low": "1/3", "high": "2/3", "mid": "1/2"}
    r = post(body)
    assert r.status_code == 422


def test_existing_validation_still_applies():
    # 沿用 events/gates/top 校验：未知引用、环、非法 p 等依旧 422
    body = make_body(
        {"a": "1/2"}, [("g", "OR", ["a", "x"])], "g",
        {"a": ("0/1", "1/1")}, "1/2")
    assert post(body).status_code == 422

    body = make_body(
        {"a": "bad"}, [], "a", {"a": ("0/1", "1/1")}, "1/2")
    assert post(body).status_code == 422


def test_node_limit_violation(monkeypatch):
    import app.main as main_mod
    monkeypatch.setattr(main_mod, "MAX_NONTERMINAL_NODES", 5)
    evs = {f"e{i}": "1/2" for i in range(8)}
    gates = []
    prev = "e0"
    for i in range(1, 8):
        o, a, n, x = f"o{i}", f"a{i}", f"n{i}", f"x{i}"
        gates.append((o, "OR", [prev, f"e{i}"]))
        gates.append((a, "AND", [prev, f"e{i}"]))
        gates.append((n, "NOT", [a]))
        gates.append((x, "AND", [o, n]))
        prev = x
    body = make_body(evs, gates, prev,
                     {"e0": ("1/4", "3/4"), "e1": ("1/4", "3/4")}, "1/2")
    r = post(body)
    assert r.status_code == 422
    assert "非终端节点" in r.json()["detail"]
