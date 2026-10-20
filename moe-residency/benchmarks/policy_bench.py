#!/usr/bin/env python
"""策略 A/B：在「偏斜路由」的合成负载上比较 5 种驻留策略。

这个脚本只依赖 policy.py（不依赖引擎），可以先在纯策略层看效果，再接真实引擎。

两档负载：
  stable    单域、路由高度局部（LRU 的舒适区 —— 用来证明"别指望策略处处都赢"）
  agentic   6 域 × 多轮、每域 = 共享热点 + 专属热点（切域会造成一批冷 miss）

跑法：
    python benchmarks/policy_bench.py
"""
from __future__ import annotations

import os
import sys
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from moe_residency import PolicyConfig, make_policy  # noqa: E402
from moe_residency.profile import DomainProfile  # noqa: E402

MODES = ["lru", "slru", "prefetch", "pinned", "pinned_prefetch"]
N_EXP = 256
TOP_K = 8


def skewed(n: int, alpha: float = 1.15) -> np.ndarray:
    w = 1.0 / (np.arange(1, n + 1) ** alpha)
    return (w / w.sum()).astype(np.float64)


def make_workload(kind: str, tokens: int = 4000, seed: int = 7):
    """返回 (weights 序列, 域标签序列)。

    stable : 固定一个偏斜分布（无切域）
    agentic: 6 域，每域 = 6 个共享热点（权重 ×2.2）+ 10 个专属热点（×2.0）
    """
    rng = np.random.default_rng(seed)
    base = skewed(N_EXP)
    if kind == "stable":
        return [base] * tokens, ["d0"] * tokens

    n_dom, shared_n, dom_n = 6, 6, 10
    pool = rng.permutation(N_EXP)
    core = pool[:shared_n]
    ws, tags = [], []
    for i in range(tokens):
        d = (i // 250) % n_dom                      # 每 250 token 换一个域
        w = base.copy()
        w[core] *= 2.2
        w[pool[shared_n + d * dom_n: shared_n + (d + 1) * dom_n]] *= 2.0
        ws.append(w / w.sum())
        tags.append(f"d{d}")
    return ws, tags


def simulate(mode: str, ws, tags, cap: int, prefetch_q: int = 12):
    cfg = PolicyConfig(capacity=cap, mode=mode, pin_frac=0.3, prefetch_q=prefetch_q)
    pol = make_policy(N_EXP, cfg)
    prof = DomainProfile(N_EXP, ema=0.9)
    rng = np.random.default_rng(99)
    prev = None
    for i, (w, d) in enumerate(zip(ws, tags)):
        if d != prev:
            prev = d
            if cfg.mode in ("prefetch", "pinned_prefetch"):
                pol.switch_domain(prof.hint(d, q=prefetch_q))
        if i % 100 == 0:
            pol.refresh_pin()
        experts = rng.choice(N_EXP, size=TOP_K, replace=False, p=w).tolist()
        prof.note(d, experts)
        for e in experts:
            pol.access(e)
    n = pol.stats.hits + pol.stats.misses
    return pol.stats.misses / n, pol.stats.prefetched, pol.stats.prefetch_used


def main() -> int:
    print(f"专家 {N_EXP} | top_k {TOP_K} | 4000 token | 快层容量 64\n")
    for kind, desc in [("stable", "单域（LRU 舒适区）"), ("agentic", "6 域 × 切域（策略该赢的场景）")]:
        ws, tags = make_workload(kind)
        print(f"【{kind}】{desc}")
        print(f"  {'策略':<18}{'停顿 miss 率':>13}{'相对 LRU':>10}{'预取(用到)':>13}")
        base = None
        for mode in MODES:
            r, pf, pfu = simulate(mode, ws, tags, cap=64)
            if base is None:
                base = r
            delta = "" if mode == "lru" else f"{(r-base)/base*100:+.1f}%"
            print(f"  {mode:<18}{r*100:>12.2f}%{delta:>10}{f'{pf}({pfu})':>13}")
        print()
    print("读法：负值 = 停顿 miss 更少 = 更快；预取会略增总 I/O（用带宽换延迟），所以也报 I/O。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
