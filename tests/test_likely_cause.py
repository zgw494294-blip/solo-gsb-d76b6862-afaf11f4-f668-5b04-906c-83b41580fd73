"""POST /v1/likely-cause：最大先验乘积赋值的语义、并列裁决、精确性与 422 校验。

用穷举真值表（枚举全部事件状态）作为参考实现做交叉验证。
"""

import itertools
import random
from fractions import Fraction

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------- 穷举真值表参考实现 ----------

def brute_likely(ev_specs, gate_specs, top, observations):
    """枚举全部事件赋值，返回 (P(C), 最优赋值 dict, P(A)/P(C))。

    observations 的键可以是事件或门；门读数按结构求值。最优赋值在满足
    C（顶事件发生且全部读数成立）的完整赋值中使独立先验乘积 P(A)
    最大；并列时按事件 id 字典序比较真假序列、false 优先——
    itertools.product 按 0(false) 在前枚举，严格更大才替换即得。
    P(C)=0 时返回 None。
    """
    ev_ids = sorted(ev_specs)
    probs = {e: Fraction(ev_specs[e]) for e in ev_ids}
    gates = {g[0]: (g[1], g[2]) for g in gate_specs}

    def evaluate(state, node):
        if node in state:
            return state[node]
        op, inputs = gates[node]
        vals = [evaluate(state, i) for i in inputs]
        if op == "AND":
            return all(vals)
        if op == "OR":
            return any(vals)
        return not vals[0]

    cond_prob = Fraction(0)
    best_w = None
    best_state = None
    for bits in itertools.product((0, 1), repeat=len(ev_ids)):
        state = dict(zip(ev_ids, (bool(b) for b in bits)))
        if not evaluate(state, top):
            continue
        if any(
            evaluate(state, oid) != value
            for oid, value in observations.items()
        ):
            continue
        weight = Fraction(1)
        for e in ev_ids:
            weight *= probs[e] if state[e] else (Fraction(1) - probs[e])
        cond_prob += weight
        if best_w is None or weight > best_w:
            best_w = weight
            best_state = state

    if cond_prob == 0:
        return None
    return cond_prob, best_state, best_w / cond_prob


def f2s(f):
    return f"{f.numerator}/{f.denominator}"


def make_body(ev_specs, gate_specs, top, observations=None):
    body = {
        "events": [{"id": k, "p": v} for k, v in ev_specs.items()],
        "gates": [
            {"id": i, "op": op, "inputs": ins}
            for i, op, ins in gate_specs
        ],
        "top": top,
    }
    if observations is not None:
        body["observations"] = observations
    return body


def check_against_brute(ev_specs, gate_specs, top, observations=None):
    """对照参考实现检查一次请求的全部字段，返回响应 JSON（P(C)=0 时为 None）。"""
    obs = observations or {}
    body = make_body(ev_specs, gate_specs, top, obs)
    r = client.post("/v1/likely-cause", json=body)
    exp = brute_likely(ev_specs, gate_specs, top, obs)
    if exp is None:
        assert r.status_code == 422, (r.text, body)
        return None
    assert r.status_code == 200, (r.text, body)
    cond_prob, best_state, posterior = exp
    data = r.json()
    assert data["condition_probability"] == f2s(cond_prob)
    assert data["assignment_posterior"] == f2s(posterior)
    assert data["assignment"] == best_state
    # assignment 按事件 id 字典序排列
    assert list(data["assignment"]) == sorted(ev_specs)
    return data


# ---------- 基本语义 ----------

MOTOR_EV = {
    "motor_overheat": "1/100",
    "coolant_loss": "2/100",
    "pump_fail": "1/200",
}
MOTOR_GATES = [
    ("cooling_fail", "OR", ["coolant_loss", "pump_fail"]),
    ("top", "OR", ["motor_overheat", "cooling_fail"]),
]


def test_no_observations_picks_most_likely_failure():
    # 无观测时 C = top；最可能的顶事件成因是单独 coolant_loss
    # （P = 99/100·2/100·199/200 = 19701/1000000），优于单独
    # motor_overheat（9751/1000000）或单独 pump_fail（4851/1000000）
    data = check_against_brute(MOTOR_EV, MOTOR_GATES, "top")
    assert data["assignment"] == {
        "motor_overheat": False,
        "coolant_loss": True,
        "pump_fail": False,
    }
    assert data["condition_probability"] == "34651/1000000"
    assert data["assignment_posterior"] == "19701/34651"


def test_observations_narrow_the_cause():
    # 观测“电机未过热但冷却失效”：最可能成因变为单独 coolant_loss
    obs = {"motor_overheat": False, "cooling_fail": True}
    data = check_against_brute(MOTOR_EV, MOTOR_GATES, "top", obs)
    assert data["assignment"] == {
        "motor_overheat": False,
        "coolant_loss": True,
        "pump_fail": False,
    }
    # P(C) = 99/100 · 249/10000 = 24651/1000000
    assert data["condition_probability"] == "24651/1000000"
    # P(A) = 99/100 · 1/50 · 199/200 = 19701/1000000
    assert data["assignment_posterior"] == "199/249"


def test_empty_observations_allowed():
    body = make_body(MOTOR_EV, MOTOR_GATES, "top", {})
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["assignment"]["coolant_loss"] is True


def test_observations_omitted_allowed():
    body = make_body({"a": "1/2"}, [], "a")
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 200, r.text
    assert r.json() == {
        "assignment": {"a": True},
        "condition_probability": "1/2",
        "assignment_posterior": "1/1",
    }


def test_tie_break_prefers_false():
    # top = a OR b，p 均为 1/2：(F,T)、(T,F)、(T,T) 并列 1/4，
    # 按事件 id 字典序比较真假序列，false 优先 => a=false, b=true
    data = check_against_brute(
        {"a": "1/2", "b": "1/2"},
        [("g", "OR", ["a", "b"])],
        "g",
    )
    assert data["assignment"] == {"a": False, "b": True}
    assert data["condition_probability"] == "3/4"
    assert data["assignment_posterior"] == "1/3"


def test_free_variables_take_more_likely_side():
    # b 与顶事件无关：p=1/3 < 1/2 取 false；c 恒真（p=1/1）取 true
    ev = {"a": "3/4", "b": "1/3", "c": "1/1"}
    data = check_against_brute(ev, [], "a")
    assert data["assignment"] == {"a": True, "b": False, "c": True}
    assert data["condition_probability"] == "3/4"
    # P(A) = 3/4 · 2/3 · 1 = 1/2
    assert data["assignment_posterior"] == "2/3"


def test_free_variable_with_half_probability_prefers_false():
    # p=1/2 的自由变量两侧权重相等，false 优先
    data = check_against_brute({"a": "1/4", "b": "1/2"}, [], "a")
    assert data["assignment"] == {"a": True, "b": False}
    # P(A) = 1/4 · 1/2 = 1/8，P(C) = 1/4
    assert data["assignment_posterior"] == "1/2"


def test_not_gate_condition():
    # top = NOT a：满足 C 的唯一赋值是 a=false
    data = check_against_brute({"a": "3/7"}, [("g", "NOT", ["a"])], "g")
    assert data["assignment"] == {"a": False}
    assert data["condition_probability"] == "4/7"
    assert data["assignment_posterior"] == "1/1"


def test_shared_references_single_assignment():
    # (a AND b) OR (a AND c)，观测 g2=false：a=b=true、c=false 成为
    # 唯一可能的顶事件成因之一，共享的 a 只取一个值
    ev = {"a": "1/2", "b": "1/3", "c": "1/5"}
    gates = [
        ("g1", "AND", ["a", "b"]),
        ("g2", "AND", ["a", "c"]),
        ("top", "OR", ["g1", "g2"]),
    ]
    data = check_against_brute(ev, gates, "top", {"g2": False})
    assert data["assignment"] == {"a": True, "b": True, "c": False}
    assert data["condition_probability"] == "2/15"
    assert data["assignment_posterior"] == "1/1"


def test_observed_unreachable_gate_still_built():
    # top 不经过 g，但观测了 g=true：条件传导到 a、b
    data = check_against_brute(
        {"a": "1/2", "b": "1/3"},
        [("g", "AND", ["a", "b"])],
        "a",
        {"g": True},
    )
    assert data["assignment"] == {"a": True, "b": True}
    assert data["condition_probability"] == "1/6"
    assert data["assignment_posterior"] == "1/1"


def test_gate_observation_false():
    # 观测与 top 无关的 OR 门为 false => 两个输入都必须为 false
    data = check_against_brute(
        {"a": "2/3", "b": "3/5", "c": "1/2"},
        [("g", "OR", ["a", "b"])],
        "c",
        {"g": False},
    )
    assert data["assignment"] == {"a": False, "b": False, "c": True}
    assert data["condition_probability"] == "1/15"
    assert data["assignment_posterior"] == "1/1"


def test_constant_true_top():
    # top = a OR NOT a 恒真：C 退化为全自由，取各事件较可能一侧
    data = check_against_brute(
        {"a": "2/3", "b": "1/2"},
        [("na", "NOT", ["a"]), ("top", "OR", ["a", "na"])],
        "top",
    )
    assert data["assignment"] == {"a": True, "b": False}
    assert data["condition_probability"] == "1/1"
    # P(A) = 2/3 · 1/2 = 1/3
    assert data["assignment_posterior"] == "1/3"


# ---------- 随机电路穷举对照 ----------

def test_random_circuits_against_truth_table():
    rng = random.Random(20260928)
    checked = 0
    while checked < 40:
        n = rng.randint(1, 10)
        ev_ids = [f"e{i}" for i in range(n)]
        ev_specs = {}
        for e in ev_ids:
            d = rng.randint(1, 8)
            ev_specs[e] = f"{rng.randint(0, d)}/{d}"
        gates = []
        available = list(ev_ids)
        for j in range(rng.randint(0, min(16, 3 * n))):
            gid = f"g{j}"
            if len(available) < 2:
                op = "NOT"
            else:
                op = rng.choice(["AND", "OR", "NOT"])
            if op == "NOT":
                inputs = [rng.choice(available)]
            else:
                k = rng.randint(2, min(8, len(available)))
                inputs = rng.sample(available, k)
            gates.append((gid, op, inputs))
            available.append(gid)
        top = rng.choice(available)

        # 0~16 个观测，键可取自事件或门
        candidates = list(ev_ids) + [g[0] for g in gates]
        k = min(len(candidates), rng.randint(0, 16))
        chosen = rng.sample(candidates, k)
        observations = {oid: rng.choice([True, False]) for oid in chosen}

        data = check_against_brute(ev_specs, gates, top, observations)
        if data is None:
            continue
        # 后验必须是合法概率
        num, den = data["assignment_posterior"].split("/")
        assert 0 < int(num) <= int(den)
        checked += 1


# ---------- P(C)=0 ----------

def test_contradictory_observations_return_422():
    # a=true 且 NOT a=true 永远不可同时成立
    body = make_body(
        {"a": "1/2"},
        [("na", "NOT", ["a"])], "na",
        {"a": True, "na": True},
    )
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "condition_probability=0" in detail and "矛盾" in detail


def test_structurally_false_top_returns_422():
    # top = a AND NOT a 恒假：顶事件不可能发生
    body = make_body(
        {"a": "1/2"},
        [("na", "NOT", ["a"]), ("top", "AND", ["a", "na"])],
        "top",
    )
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "condition_probability=0" in detail and "矛盾" in detail


def test_zero_prior_event_required_true_returns_422():
    # 结构可满足（a=1 使 top 成立），但 P(a)=0
    body = make_body({"a": "0/1"}, [], "a")
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "condition_probability=0" in detail and "先验" in detail


def test_zero_prior_event_required_true_by_observation_returns_422():
    body = make_body({"a": "0/1", "b": "1/2"}, [], "b", {"a": True})
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 422
    assert "condition_probability=0" in r.json()["detail"]


# ---------- 422 校验路径 ----------

def test_unknown_observation_key():
    body = make_body({"a": "1/2"}, [], "a", {"a": True, "nope": False})
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 422
    assert "observations" in r.json()["detail"] and "nope" in r.json()["detail"]


@pytest.mark.parametrize("bad_value", [1, 0, "true", None, 1.0])
def test_non_boolean_observation_value_rejected(bad_value):
    body = make_body({"a": "1/2"}, [], "a", {"a": bad_value})
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 422


def test_too_many_observations_rejected():
    ev_specs = {f"e{i}": "1/2" for i in range(17)}
    body = make_body(
        ev_specs, [], "e0",
        {f"e{i}": True for i in range(17)},
    )
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 422


def test_sixteen_observations_allowed():
    ev_specs = {f"e{i}": "1/2" for i in range(16)}
    body = make_body(
        ev_specs, [], "e0",
        {f"e{i}": True for i in range(16)},
    )
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert all(v is True for v in data["assignment"].values())
    assert data["assignment_posterior"] == "1/1"


def test_extra_field_rejected():
    r = client.post("/v1/likely-cause", json={
        "events": [{"id": "a", "p": "1/2"}],
        "gates": [], "top": "a",
        "observations": {"a": True},
        "extra": 1})
    assert r.status_code == 422


def test_original_validation_still_applies():
    # 未知 top
    body = make_body({"a": "1/2"}, [], "nope")
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 422 and "顶层" in r.json()["detail"]
    # 环
    body = make_body({"a": "1/2"}, [("g", "OR", ["a", "g"])], "g")
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 422 and "环" in r.json()["detail"]
    # 非法概率
    body = make_body({"a": "3/2"}, [], "a")
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 422 and "概率" in r.json()["detail"]
    # 非 ASCII id
    body = make_body({"事件": "1/2"}, [], "事件")
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 422


def test_node_limit_in_condition_conjunction(monkeypatch):
    # 树本身节点不多，但合取观测约束可能新增节点；调小上限后应返回 422
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
    body = make_body(evs, gates, prev, {prev: True})
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 422
    assert "非终端节点" in r.json()["detail"]


# ---------- 结构矛盾优先于节点额度 ----------

def _xor_chain_gates(prefix, first, ids):
    """first XOR ids[0] XOR ids[1] ...；每步 4 门（OR/AND/NOT/AND）。

    返回 (门列表, 链顶 id)。该链取反在共享唯一表中是新鲜节点，
    可把共享准备阶段的累计节点数推高。
    """
    gates = []
    prev = first
    for i, nxt in enumerate(ids, start=1):
        o = f"{prefix}_o{i}"
        a = f"{prefix}_a{i}"
        n = f"{prefix}_na{i}"
        x = f"{prefix}_x{i}"
        gates.append((o, "OR", [prev, nxt]))
        gates.append((a, "AND", [prev, nxt]))
        gates.append((n, "NOT", [a]))
        gates.append((x, "AND", [o, n]))
        prev = x
    return gates, prev


def _near_budget_tree():
    """合法双 XOR 链汇合门 and13、top=and13 OR 末事件。

    走真实建图路径时累计 487 个非终端节点；为 and13 预先取反将增至
    504。把额度调为 490 即复现“建图接近上限、复杂 false 读数取反
    超限”的现场（线上为接近 50000 的同构情形）。
    """
    evs = {f"e{i}": "1/2" for i in range(16)}
    g1, o1 = _xor_chain_gates("lx", "e0", [f"e{i}" for i in range(1, 13)])
    g2, o2 = _xor_chain_gates("ly", "e13", ["e14", "e15"])
    gates = g1 + g2 + [
        ("and13", "AND", [o1, o2]),
        ("top", "OR", ["and13", "e15"]),
    ]
    return evs, gates


def test_structural_contradiction_with_top_false_precedes_node_limit(
    monkeypatch,
):
    # top=false 与 and13=false 同时给出：C=top∧E 必含 top∧¬top，结构恒假。
    # 即使共享准备过程为 and13 预先取反就会越过额度，也必须返回结构矛盾
    # 422，而不是节点超限。
    import app.main as main_mod
    monkeypatch.setattr(main_mod, "MAX_NONTERMINAL_NODES", 490)
    evs, gates = _near_budget_tree()
    body = make_body(evs, gates, "top", {"and13": False, "top": False})
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert "矛盾" in detail and "condition_probability=0" in detail
    assert "非终端节点" not in detail


def test_top_false_alone_is_structural_contradiction_near_limit(monkeypatch):
    # 无关读数不得改变错误原因：去掉复杂的 and13 读数后，top=false 单独
    # 仍返回同一类结构矛盾 422。
    import app.main as main_mod
    monkeypatch.setattr(main_mod, "MAX_NONTERMINAL_NODES", 490)
    evs, gates = _near_budget_tree()
    body = make_body(evs, gates, "top", {"top": False})
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "矛盾" in detail and "非终端节点" not in detail


def test_combined_readings_negating_top_precede_node_limit(monkeypatch):
    # 多条读数合起来才否定顶事件：top=and13 OR e15，and13=false 与
    # e15=false 联立即 top=false（单看任一读数都不与 top 矛盾），
    # 另加的 e13=false 是强制文字。复杂读数 and13=false 的取反本会越过
    # 额度，但结构矛盾必须在构建任何取反/合取节点前判定，返回 422
    # 结构矛盾而非节点超限。
    import app.main as main_mod
    monkeypatch.setattr(main_mod, "MAX_NONTERMINAL_NODES", 490)
    evs, gates = _near_budget_tree()
    body = make_body(
        evs, gates, "top",
        {"and13": False, "e13": False, "e15": False},
    )
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert "矛盾" in detail and "condition_probability=0" in detail
    assert "非终端节点" not in detail


def test_combined_contradiction_independent_of_reading_order(monkeypatch):
    # 结论不受读数给出顺序影响：JSON 对象顺序倒置后仍是同一结构矛盾
    import app.main as main_mod
    monkeypatch.setattr(main_mod, "MAX_NONTERMINAL_NODES", 490)
    evs, gates = _near_budget_tree()
    for obs in (
        {"e15": False, "e13": False, "and13": False},
        {"e13": False, "e15": False, "and13": False},
        {"and13": False, "e15": False},
    ):
        body = make_body(evs, gates, "top", obs)
        r = client.post("/v1/likely-cause", json=body)
        assert r.status_code == 422, (obs, r.text)
        detail = r.json()["detail"]
        assert "矛盾" in detail and "非终端节点" not in detail, obs


def test_combined_contradiction_unaffected_by_irrelevant_reading(monkeypatch):
    # 追加与矛盾无关的读数（可满足的任意门读数）不改变错误原因
    import app.main as main_mod
    monkeypatch.setattr(main_mod, "MAX_NONTERMINAL_NODES", 490)
    evs, gates = _near_budget_tree()
    for obs in (
        {"and13": False, "e15": False, "e14": True},
        {"and13": False, "e15": False, "e14": False},
        {"and13": False, "e15": False, "e0": True, "e1": False},
    ):
        body = make_body(evs, gates, "top", obs)
        r = client.post("/v1/likely-cause", json=body)
        assert r.status_code == 422, (obs, r.text)
        assert "矛盾" in r.json()["detail"] and "非终端节点" not in r.json()["detail"]


def test_t_equiv_c_or_z_with_both_false_is_contradiction():
    # 任务现场的小型同构：t = c OR z，读数 {c:false, z:false} 联立否定 t
    evs = {"c": "1/2", "z": "1/2"}
    gates = [("t", "OR", ["c", "z"])]
    # 两条都给
    r = client.post(
        "/v1/likely-cause", json=make_body(evs, gates, "t", {"c": False, "z": False})
    )
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "矛盾" in detail and "condition_probability=0" in detail
    # 删掉任一条读数则可满足，正常返回成因
    for obs in ({"c": False}, {"z": False}):
        data = check_against_brute(evs, gates, "t", obs)
        assert data is not None


def test_satisfiable_near_limit_still_returns_normal_result(monkeypatch):
    # 结构可满足时限额与正常成因字段不变：and13=true 蕴含 top。
    # 注意该电路穷举求值随门深度指数增长，故此处直接断言精确结果而不
    # 走 check_against_brute（其递归求值器无记忆化）：
    # 两条 13 位/3 位 XOR 链各 1/2 奇校验 => P(and13)=1/4 => P(C)=1/4；
    # 全部 p=1/2 时 16384 组满足赋值等权 1/65536，按 id 字典序 false
    # 优先取真假序列最小者。注意事件按 id 字典序排列（e0,e1,e10,…），
    # 字典序最小的奇校验组合只置 e9 与 e15 为 true。
    import app.main as main_mod
    monkeypatch.setattr(main_mod, "MAX_NONTERMINAL_NODES", 490)
    evs, gates = _near_budget_tree()
    body = make_body(evs, gates, "top", {"and13": True})
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert list(data["assignment"]) == sorted(evs)
    assert data["condition_probability"] == "1/4"
    assert data["assignment_posterior"] == "1/16384"
    assert data["assignment"] == {
        **{f"e{i}": False for i in range(16)},
        "e9": True,
        "e15": True,
    }


def test_genuine_computation_near_limit_still_reports_node_limit(monkeypatch):
    # 不与 top 矛盾、确需计算的复杂 false 读数（and13=false）超限时，
    # 仍返回节点超限 422，不被结构矛盾短路吞掉。
    import app.main as main_mod
    monkeypatch.setattr(main_mod, "MAX_NONTERMINAL_NODES", 490)
    evs, gates = _near_budget_tree()
    body = make_body(evs, gates, "top", {"and13": False})
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 422
    assert "非终端节点" in r.json()["detail"]


def test_diagnose_still_conjoins_full_evidence_near_limit(monkeypatch):
    # /v1/diagnose 共用准备过程但仍按完整 E 计算（先为 and13 取反）：
    # 该额度下其取反越过额度，返回节点超限——与 likely-cause 的结构矛盾
    # 路径相互独立，本修复不改变 diagnose 的原语义。
    import app.main as main_mod
    monkeypatch.setattr(main_mod, "MAX_NONTERMINAL_NODES", 490)
    evs, gates = _near_budget_tree()
    body = {
        "events": [{"id": k, "p": v} for k, v in evs.items()],
        "gates": [
            {"id": i, "op": op, "inputs": ins} for i, op, ins in gates
        ],
        "top": "top",
        "observations": {"and13": False, "top": False},
    }
    r = client.post("/v1/diagnose", json=body)
    assert r.status_code == 422
    assert "非终端节点" in r.json()["detail"]
