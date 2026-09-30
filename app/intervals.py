"""概率区间端点极值：在已构建的 ROBDD 上求顶事件概率关于若干事件概率的
稀疏多线性多项式，并在各区间端点张成的超立方体顶点上精确取最小/最大值。

顶事件概率 P 关于各独立事件概率是多线性函数（ROBDD 中每个变量在每条
根到终端的路径上至多被决定一次，节点递推
P(n) = (1-p_v)·P(low) + p_v·P(high) 对 p_v 只做一次线性组合），故它在
每个区间变量 x_j∈[low_j, high_j] 上单独都是线性的——逐个变量代入端点
即可把任一内部点归约到某个顶点，因此超立方体上的最小值与最大值必在
端点组合（2^k 种，k≤8，至多 256 种）处取得，无需枚举 2^n 个事件状态。

自底向上为每个 BDD 节点维护关于 k 个区间变量的稀疏多线性多项式，以 k
位掩码指示单项式包含哪些区间变量（区间变量按事件 id 字典序排位）：
  - 概率固定的变量 var：
        Q(S) = (1-p_var)·Q_low(S) + p_var·Q_high(S)；
  - 区间变量 j（对应位 b=1<<j）：
        Q = (1-x_j)·Q_low + x_j·Q_high
          = Q_low + x_j·(Q_high − Q_low)，
    即 Q(S)（S 不含 b）= Q_low(S)，
       Q(S∪b)        = Q_high(S) − Q_low(S)。
被 ROBDD 约简跳过的变量对纯概率的贡献恒为 ((1-p)+p)=1，无需补因子。

终端 0 为零多项式（空字典），终端 1 为 {0: 1}。子节点 id 总是小于父节点
id，按 id 递增即为后序遍历；子节点多项式在被全部父节点用完后即释放
（根额外保留一次），内存只与遍历前沿有关（同 bdd.count_distribution）。

顶点评估：P(mask) = Σ_S c_S · Π_{j∈S} x_j，掩码第 j 位为 1 表示 x_j 取
high，否则取 low；未在顶事件 BDD 中出现的区间变量不进入任何单项式，
其端点选择不影响概率，按并列规则取 low。

并列（同一极值被多个端点组合取得）时按事件 id 字典序逐位比较端点序列，
low（位为 0）优先：第 0 位对应字典序最小的区间事件，两个掩码在最低分歧
位上为 0 者数值更小，故取数值最小的掩码即可。
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Tuple

from .bdd import ROBDD

# (最小概率, 最小端点掩码, 最大概率, 最大端点掩码)
Extrema = Tuple[Fraction, int, Fraction, int]


def probability_polynomial(
    bdd: ROBDD,
    root: int,
    probs: List[Fraction],
    interval_vars: List[int],
) -> Dict[int, Fraction]:
    """求顶事件概率关于区间变量的稀疏多线性多项式 {掩码: 系数}。

    interval_vars 为区间事件的变量下标（按事件 id 字典序排列）；其在列表
    中的位置即单项式掩码的位序。根为终端 0 时返回空字典（零多项式），
    终端 1 时返回 {0: 1}。
    """
    bit_of = {var: i for i, var in enumerate(interval_vars)}

    # 节点多项式：终端 0 为 {}，终端 1 为 {0: 1}
    polys: List[Dict[int, Fraction] | None] = [None] * len(bdd.nodes)
    polys[0] = {}
    polys[1] = {0: Fraction(1)}

    # 引用计数：子节点多项式被全部父节点用完后即可释放
    uses = [0] * len(bdd.nodes)
    for nid in range(2, len(bdd.nodes)):
        node = bdd.nodes[nid]
        assert node is not None
        uses[node[1]] += 1
        uses[node[2]] += 1
    uses[root] += 1

    for nid in range(2, len(bdd.nodes)):
        node = bdd.nodes[nid]
        assert node is not None
        var, low, high = node
        pl = polys[low]
        ph = polys[high]
        assert pl is not None and ph is not None

        if var in bit_of:
            bit = 1 << bit_of[var]
            # Q_low 落在不含 bit 的掩码上
            out: Dict[int, Fraction] = dict(pl)
            # Q_high − Q_low 落在含 bit 的掩码上
            for mask, coef in ph.items():
                key = mask | bit
                out[key] = out.get(key, Fraction(0)) + coef
            for mask, coef in pl.items():
                key = mask | bit
                out[key] = out.get(key, Fraction(0)) - coef
        else:
            # 固定概率变量：两个子多项式按 (1-p)、p 加权合并
            p_var = probs[var]
            q_var = Fraction(1) - p_var
            out = {}
            for mask, coef in pl.items():
                out[mask] = q_var * coef
            for mask, coef in ph.items():
                out[mask] = out.get(mask, Fraction(0)) + p_var * coef

        # 丢弃零系数以保持稀疏（相消后可能为 0）
        polys[nid] = {mask: coef for mask, coef in out.items() if coef != 0}

        for child in (low, high):
            if child >= 2:
                uses[child] -= 1
                if uses[child] == 0:
                    polys[child] = None

    root_poly = polys[root]
    assert root_poly is not None
    return root_poly


def probability_extrema(
    bdd: ROBDD,
    root: int,
    probs: List[Fraction],
    interval_vars: List[int],
    lows: List[Fraction],
    highs: List[Fraction],
) -> Extrema:
    """求顶事件概率在区间端点组合上的 (最小值, 最小掩码, 最大值, 最大掩码)。

    lows/highs 与 interval_vars 对齐（均按事件 id 字典序）；掩码第 j 位为
    1 表示第 j 个区间事件取 high，为 0 表示取 low。并列时按事件 id 字典序
    比较端点、low 优先，即取数值最小的掩码。
    """
    coeffs = probability_polynomial(bdd, root, probs, interval_vars)
    k = len(interval_vars)

    min_value = max_value = Fraction(0)
    min_mask = max_mask = 0
    found = False
    # 掩码按数值递增遍历，严格更优才替换：并列时自然保留最小掩码
    for vertex in range(1 << k):
        total = Fraction(0)
        for mask, coef in coeffs.items():
            monomial = Fraction(1)
            rest = mask
            while rest:
                bit = rest & -rest
                j = bit.bit_length() - 1
                monomial *= highs[j] if (vertex >> j) & 1 else lows[j]
                rest -= bit
            total += coef * monomial
        if not found:
            min_value = max_value = total
            min_mask = max_mask = vertex
            found = True
            continue
        if total < min_value:
            min_value, min_mask = total, vertex
        if total > max_value:
            max_value, max_mask = total, vertex

    return min_value, min_mask, max_value, max_mask
