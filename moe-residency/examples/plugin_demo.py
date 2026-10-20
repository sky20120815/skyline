#!/usr/bin/env python
"""以**插件**方式驱动一个引擎的端到端演示。

展示的东西：
  1. 宿主引擎只需 3 个调用点（before_route / on_route / after_token）
  2. 插件负责：切域预取、命中统计、驻留集合对齐、逐位验证
  3. 走标准 entry point 发现（未安装时回退到直接 import）

跑法：
    python examples/plugin_demo.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from moe_residency import BitExactVerifier, ResidencyPlugin  # noqa: E402
from moe_residency.engine import ReferenceMoE, TieredExpertStore  # noqa: E402

DOMAINS = ["code", "chat", "math"]


def discover_plugin():
    """标准插件发现：entry point → 直接 import（开发态回退）。"""
    try:
        from importlib.metadata import entry_points
        eps = [e for e in entry_points(group="moe_residency.plugins")
               if e.name == "moe-residency"]
        if eps:
            print(f"  通过 entry point 发现插件：{eps[0].name} = {eps[0].value}")
            return eps[0].load()
    except Exception as exc:                                   # noqa: BLE001
        print(f"  entry point 不可用（{exc}），回退到直接 import")
    from moe_residency.plugin import load_plugin
    print("  通过直接 import 加载插件（开发态）")
    return load_plugin


def build_workload(moe, per_domain=40, cycles=3, seed=11):
    rng = np.random.default_rng(seed)
    pool = np.arange(moe.E)
    rng.shuffle(pool)
    core, biases = pool[:6], {}
    for k, name in enumerate(DOMAINS):
        b = np.zeros(moe.E, dtype=np.float32)
        b[core] += np.float32(1.1)
        b[pool[6 + k * 8: 6 + (k + 1) * 8]] += np.float32(1.0)
        biases[name] = b
    xs, tags = [], []
    for _ in range(cycles):
        for name in DOMAINS:
            base = rng.normal(0, 1, moe.d).astype(np.float32)
            x = base.copy()
            for _ in range(per_domain):
                x = (np.float32(0.5) * x + np.float32(0.5) * base
                     + np.float32(0.9) * rng.normal(0, 1, moe.d).astype(np.float32))
                xs.append(x.astype(np.float32))
                tags.append(name)
    return biases, xs, tags


def main() -> int:
    print("=" * 74)
    print("以插件方式驱动引擎：宿主只需 3 个调用点")
    print("=" * 74)
    loader = discover_plugin()

    moe = ReferenceMoE(n_experts=64, d_model=32, hidden=64, top_k=4,
                       seed=1337, router_skew=0.2)
    biases, xs, tags = build_workload(moe)

    # ---------- 宿主 A：全驻留（golden） ----------
    ver = BitExactVerifier()
    ver.start("resident")
    for x, dom in zip(xs, tags):
        ver.record("resident", moe.forward(x, domain_bias=biases[dom])[0])

    # ---------- 宿主 B：三级分层 + 插件 ----------
    plugin = loader(n_experts=moe.E, capacity=16, mode="prefetch",
                    prefetch_q=8, verifier=ver, run_name="tiered")
    tmp = tempfile.mkdtemp(prefix="moe_disk_")
    store = TieredExpertStore(moe.blocks, warm_cap=20, disk_dir=tmp)

    for x, dom in zip(xs, tags):
        # ① 切任务时（路由之前）：插件告诉我们该预取谁
        prefetch = plugin.before_route(domain=dom)
        for eid in prefetch:                       # 宿主负责把专家搬进快层
            store.put(eid, store.fetch(eid))
        # ② 路由之后：插件告诉我们哪些 miss 需要搬
        out, _ = moe.forward(x, policy=None, store=None,
                             domain=dom, domain_bias=biases[dom])
        ids = _route_ids(moe, x, biases[dom])
        misses = plugin.on_route(ids)
        for eid, missed in misses.items():
            if missed:
                store.put(eid, store.fetch(eid))
        # ③ token 结束：对齐驻留集合
        store.retain(plugin.policy.resident())
        plugin.after_token()
        ver.record("tiered", out)

    ok, info = ver.compare("resident", "tiered")
    print(f"\n逐位不变：{'✅ 通过' if ok else '❌ 不一致'}"
          f"（{info['steps']} token，不一致 {info['mismatched_steps']} 步）")
    print(f"磁盘目录：{tmp}")
    print("\n插件统计：")
    for k, v in plugin.stats().items():
        print(f"  {k:<12} = {v}")
    return 0 if ok else 1


def _route_ids(moe: ReferenceMoE, x: np.ndarray, bias: np.ndarray):
    """复现引擎的路由（宿主自己的事，插件不参与）。"""
    x = np.asarray(x, dtype=np.float32).reshape(moe.d)
    mu = np.float32(x.mean())
    sd = np.float32(np.sqrt(np.float32(x.var()) + np.float32(1e-5)))
    h = (x - mu) / sd * moe.norm_g + moe.norm_b
    h = h @ moe.W_in
    r = h @ moe.router + bias
    return np.argsort(-r, kind="stable")[: moe.k].tolist()


if __name__ == "__main__":
    raise SystemExit(main())
