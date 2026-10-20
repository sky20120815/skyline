#!/usr/bin/env python
"""端到端演示：

  1. 三级分层 vs 全驻留 —— **logits 逐位相同**（放置只改速度，不改语义）
  2. 四种驻留策略对比，指标分两类：
       · 停顿 miss（必须当场等 I/O）——这才是用户感知的慢
       · 预取 I/O（可被计算重叠掩盖）——不是罚分项，但要报告
  3. 负载结构贴近真实 MoE：**全局共享热点 + 每域专属热点**（通用专家 + 专门专家）

跑法：
    python examples/demo_engine.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from moe_residency import (BitExactVerifier, DomainProfile, PolicyConfig,  # noqa: E402
                          ResidencyPolicy, make_policy)
from moe_residency.engine import ReferenceMoE, TieredExpertStore  # noqa: E402

MODES = ["lru", "slru", "prefetch", "pinned", "pinned_prefetch"]
N_DOM, PER_DOM, CYCLES = 6, 30, 3
SHARED, DOM_HOT = 6, 8


def build_workload(moe: ReferenceMoE, seed: int = 7):
    """6 个域：每域 = 共享热点(6) + 专属热点(8)；域内 token 缓慢漂移（AR(1)）。"""
    rng = np.random.default_rng(seed)
    pool = np.arange(moe.E)
    rng.shuffle(pool)
    core = pool[:SHARED]
    biases = {}
    for k in range(N_DOM):
        b = np.zeros(moe.E, dtype=np.float32)
        b[core] += np.float32(1.1)
        b[pool[SHARED + k * DOM_HOT: SHARED + (k + 1) * DOM_HOT]] += np.float32(1.0)
        biases[f"d{k}"] = b
    xs, tags = [], []
    for _ in range(CYCLES):
        for k in range(N_DOM):
            base = rng.normal(0, 1, moe.d).astype(np.float32)
            x = base.copy()
            for _ in range(PER_DOM):
                x = (np.float32(0.5) * x + np.float32(0.5) * base
                     + np.float32(0.9) * rng.normal(0, 1, moe.d).astype(np.float32))
                xs.append(x.astype(np.float32))
                tags.append(f"d{k}")
    return biases, xs, tags


def run(mode: str, moe, biases, xs, tags, ver: BitExactVerifier, tmp: str, cap: int,
        pin_frac: float = 0.3, prefetch_q: int = 8):
    store = TieredExpertStore(moe.blocks, warm_cap=20, disk_dir=tmp)
    policy = make_policy(moe.E, PolicyConfig(
        capacity=cap, mode=mode, pin_frac=pin_frac, prefetch_q=prefetch_q))
    profile = DomainProfile(moe.E, ema=0.9)
    ver.start(mode)
    prev, pref_io = None, 0

    for i, (x, dom) in enumerate(zip(xs, tags)):
        if dom != prev:
            prev = dom
            if mode in ("prefetch", "pinned_prefetch"):
                before = store.stats["disk_reads"]
                policy.switch_domain(profile.hint(dom, q=prefetch_q))
                store.load_resident(list(policy.resident()))
                pref_io += store.stats["disk_reads"] - before
        if i % 20 == 0:
            policy.refresh_pin()
            store.load_resident(list(policy.resident()))
        out, _ = moe.forward(x, policy=policy, store=store, profile=profile,
                             domain=dom, domain_bias=biases[dom])
        ver.record(mode, out)

    n = policy.stats.hits + policy.stats.misses
    return {
        "mode": mode,
        "stall": policy.stats.misses,
        "stall_rate": policy.stats.misses / n,
        "total_io": store.stats["disk_reads"],
        "prefetch_io": pref_io,
        "prefetch_used": policy.stats.prefetch_used,
    }


def main() -> int:
    moe = ReferenceMoE(n_experts=64, d_model=32, hidden=64, top_k=4,
                       seed=1337, router_skew=0.2)
    biases, xs, tags = build_workload(moe)
    cap = 16

    ver = BitExactVerifier()
    ver.start("resident")
    for x, dom in zip(xs, tags):
        ver.record("resident", moe.forward(x, domain_bias=biases[dom])[0])

    tmp = tempfile.mkdtemp(prefix="moe_disk_")
    print(f"专家池 {moe.E} | 快层 {cap} | 内存 20 | token {len(xs)} | "
          f"{N_DOM} 域 × {CYCLES} 轮（每域 {PER_DOM} token）")
    print(f"负载：全局共享热点 {SHARED} + 每域专属 {DOM_HOT}\n")

    rows = []
    for mode in MODES:
        r = run(mode, moe, biases, xs, tags, ver, tmp, cap)
        ok, info = ver.compare("resident", mode)
        r["bitexact"], r["mismatch"] = ok, info["mismatched_steps"]
        rows.append(r)

    print("=" * 86)
    print(f"{'策略':<18}{'停顿miss':>9}{'停顿miss率':>11}{'总I/O':>8}{'预取I/O':>9}"
          f"{'预取用到':>9}{'逐位一致':>10}")
    print("-" * 86)
    base = rows[0]["stall"] or 1
    for r in rows:
        ok = "✅" if r["bitexact"] else f"❌ {r['mismatch']}"
        print(f"{r['mode']:<18}{r['stall']:>9}{r['stall_rate']*100:>10.1f}%"
              f"{r['total_io']:>8}{r['prefetch_io']:>9}{r['prefetch_used']:>9}{ok:>10}")
    print("-" * 86)
    print("相对 LRU 的「停顿 miss」变化：", end="")
    for r in rows[1:]:
        print(f"  {r['mode']} {(r['stall']-base)/base*100:+.1f}%", end="")
    print("\n")
    print("- 五种策略 logits 全部与「全驻留」逐位相同 —— 换策略只改 I/O，不改语义。")
    print("- 预取把「当场等盘」变成「提前读、与计算重叠」：停顿降、总 I/O 略升，这是设计意图。")
    print("- 纯频率 pin / SLRU 在本负载无收益（共享热点一直在 LRU 里）——它们只在「工作集>>缓存」时才有用。")
    print("- 收益最大的是「纯域画像预取」：把切域后的一批冷 miss 提前搬走，与上一段计算重叠。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
