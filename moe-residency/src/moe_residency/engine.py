"""参考引擎：三级专家存储 + 单块 MoE 的确定性前向。

⚠ 这是一个**玩具引擎**（~200 行，numpy），用途是把"分层放置 + 逐位不变"跑通并可验证，
   不追求跑真实大模型。真实部署请接 colibri / FreeToken / llama.cpp（见 docs/INTEGRATION.md）。

为什么只做"单块 MoE"：每 token 只需 top_k 个专家 ⇒ 三层之间只搬 4 次（而非 60 层 × 8 次），
这也是本项目架构对磁盘层天然友好的原因（详见 docs/DESIGN.md）。

设计约定（很重要）：**驻留集合由 policy 唯一决定**，store 只负责"物化"和"释放"。
这样 hit/miss 统计与真实驻留永远一致，不会出现两套 LRU 打架。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


# ----------------------------------------------------------------------------
# 三级存储
# ----------------------------------------------------------------------------
class TieredExpertStore:
    """fast(VRAM 模拟) / warm(RAM 模拟) / cold(磁盘，真文件) 三级专家存储。

    `fast` 里放什么由 `policy.resident()` 决定（`retain()` 对齐）。
    `warm` 可设上限（模拟内存装不下全池）；超限的部分落 `disk_dir`（真读盘）。
    """

    def __init__(self, blocks: Dict[int, Dict[str, np.ndarray]],
                 warm_cap: Optional[int] = None, disk_dir: Optional[str] = None) -> None:
        self.warm_cap = None if warm_cap is None else int(warm_cap)
        self.disk_dir = disk_dir
        self.fast: Dict[int, Dict[str, np.ndarray]] = {}
        self.warm: Dict[int, Dict[str, np.ndarray]] = {}
        self.paths: Dict[int, str] = {}
        self.stats = {"disk_reads": 0, "warm_hits": 0, "promotions": 0, "releases": 0}
        self._blocks_ref = blocks

        if disk_dir:
            os.makedirs(disk_dir, exist_ok=True)
            for eid, blk in blocks.items():
                p = os.path.join(disk_dir, f"e{int(eid):05d}.npz")
                np.savez(p, **{k: np.asarray(v, dtype=np.float32) for k, v in blk.items()})
                self.paths[int(eid)] = p
        # 注意：即使有磁盘层，RAM 也可以缓存一部分（由 warm_cap 决定）
        for eid, blk in blocks.items():
            self._put_warm(int(eid), blk)

    def _put_warm(self, eid: int, blk: Dict[str, np.ndarray]) -> None:
        if eid in self.warm:
            return
        if self.warm_cap is not None and len(self.warm) >= self.warm_cap:
            return                      # 内存满 ⇒ 该专家只存在于磁盘
        self.warm[eid] = blk

    # ------------------------------------------------------------------ 取用
    def fetch(self, eid: int) -> Dict[str, np.ndarray]:
        """从 warm 或磁盘取一块（**不查 fast** —— fast 由 policy 管）。"""
        eid = int(eid)
        if eid in self.warm:
            self.stats["warm_hits"] += 1
            return self.warm[eid]
        if eid in self.paths:
            self.stats["disk_reads"] += 1
            with np.load(self.paths[eid]) as z:
                return {k: z[k] for k in z.files}
        # warm 上限把块挤掉了但没写盘 ⇒ 回退到内存原表（仅测试便利）
        return self._blocks_ref[eid]

    def put(self, eid: int, blk: Dict[str, np.ndarray]) -> None:
        self.fast[int(eid)] = blk
        self.stats["promotions"] += 1

    def retain(self, keep: Set[int]) -> int:
        """只保留 `keep` 里的专家在 fast 层，其余释放。返回释放数。"""
        drop = [e for e in self.fast if e not in keep]
        for e in drop:
            del self.fast[e]
        self.stats["releases"] += len(drop)
        return len(drop)

    # ------------------------------------------------------------------ 物化
    def load_resident(self, ids: Sequence[int]) -> None:
        """把一批专家放进 fast（用于 warmup / 预取 / pin 生效时）。"""
        for eid in ids:
            eid = int(eid)
            if eid not in self.fast:
                self.fast[eid] = self.fetch(eid)


# ----------------------------------------------------------------------------
# 单块 MoE（确定性）
# ----------------------------------------------------------------------------
@dataclass
class TokenTrace:
    experts: List[int] = field(default_factory=list)
    missed: List[bool] = field(default_factory=list)

    @property
    def misses(self) -> int:
        return sum(1 for m in self.missed if m)


class ReferenceMoE:
    """单块稀疏 MoE：共享主干 + E 个专家 + top-k 路由。全部 float32、确定性。"""

    def __init__(self, n_experts: int = 64, d_model: int = 32, hidden: int = 64,
                 top_k: int = 4, seed: int = 1337, router_skew: float = 1.2) -> None:
        rng = np.random.default_rng(seed)
        self.E, self.d, self.h, self.k = n_experts, d_model, hidden, top_k
        self.norm_g = rng.normal(1.0, 0.02, d_model).astype(np.float32)
        self.norm_b = rng.normal(0.0, 0.02, d_model).astype(np.float32)
        self.W_in = rng.normal(0, 0.08, (d_model, d_model)).astype(np.float32)
        self.router = rng.normal(0, 0.10, (d_model, n_experts)).astype(np.float32)
        if router_skew > 0:
            # 给路由加 Zipf 偏置 ⇒ 少数专家天然更热（贴近真实 MoE 的 power-law 路由）
            idx = np.arange(1, n_experts + 1, dtype=np.float32)
            bias = (1.0 / idx ** 1.1).astype(np.float32)
            self.router += np.float32(router_skew) * bias[None, :]
        self.W_out = rng.normal(0, 0.08, (d_model, d_model)).astype(np.float32)
        self.blocks: Dict[int, Dict[str, np.ndarray]] = {}
        for e in range(n_experts):
            self.blocks[e] = {
                "w1": rng.normal(0, 0.09, (d_model, hidden)).astype(np.float32),
                "b1": rng.normal(0, 0.01, hidden).astype(np.float32),
                "w2": rng.normal(0, 0.09, (hidden, d_model)).astype(np.float32),
                "b2": rng.normal(0, 0.01, d_model).astype(np.float32),
            }

    # ------------------------------------------------------------------ 前向
    def forward(self, x: np.ndarray, policy=None,
                store: Optional[TieredExpertStore] = None,
                profile=None, domain: Optional[str] = None,
                domain_bias: Optional[np.ndarray] = None) -> Tuple[np.ndarray, TokenTrace]:
        """一个 token 的前向。`policy=None` ⇒ 全驻留（baseline，作为逐位 golden）。

        `domain_bias`：给路由器加的按域偏置（长度 E）。真实 MoE 里"不同任务偏爱不同专家"
        就是这种形状；有了它，域画像预取才有可学的东西。
        """
        x = np.asarray(x, dtype=np.float32).reshape(self.d)
        mu = np.float32(x.mean())
        sd = np.float32(np.sqrt(np.float32(x.var()) + np.float32(1e-5)))
        h = (x - mu) / sd * self.norm_g + self.norm_b
        h = h @ self.W_in

        r = h @ self.router
        if domain_bias is not None:
            r = r + np.asarray(domain_bias, dtype=np.float32)
        order = np.argsort(-r, kind="stable")[: self.k]
        gates = _softmax(r[order])

        if profile is not None and domain is not None:
            profile.note(domain, order.tolist())

        trace = TokenTrace()
        if policy is not None and store is not None:
            for eid in order.tolist():
                hit = policy.access(eid)
                if not hit:
                    store.put(eid, store.fetch(eid))
                trace.experts.append(int(eid))
                trace.missed.append(not hit)
            store.retain(policy.resident())      # 驻留集合以 policy 为准

        acc = np.zeros(self.d, dtype=np.float32)
        for w, eid in zip(gates, order.tolist()):
            blk = self.blocks[eid] if store is None else store.fast.get(eid)
            if blk is None:                       # 兜底（正常不应发生）
                blk = self.blocks[eid]
            y = _gelu(h @ blk["w1"] + blk["b1"])
            acc = acc + np.float32(w) * (y @ blk["w2"] + blk["b2"])
        return (acc @ self.W_out, trace)

    def logits(self, x: np.ndarray) -> np.ndarray:
        """全驻留的 logits（逐位对照的 golden 路径）。"""
        return self.forward(x, policy=None, store=None)[0]


def _softmax(x: np.ndarray) -> np.ndarray:
    z = (x - x.max()).astype(np.float32)
    e = np.exp(z).astype(np.float32)
    return (e / e.sum()).astype(np.float32)


def _gelu(x: np.ndarray) -> np.ndarray:
    c = np.float32(np.sqrt(2.0 / np.pi))
    return (np.float32(0.5) * x *
            (np.float32(1.0) + np.tanh(c * (x + np.float32(0.044715) * x ** 3)))).astype(np.float32)
