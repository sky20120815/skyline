"""专家驻留策略：LRU / LFRU / 频率常驻 / 频率常驻 + 域画像预取。

与真实引擎对接时只需要两个调用点：
  · `access(eid)` -> True=命中快层 / False=未命中（需要从慢层取）
  · `switch_domain(hot_ids)` -> 返回应预取的专家列表（在路由发生**之前**调用）

四种模式：
  lru              纯 LRU（llama.cpp / FreeToken 的基线）
  lfru             频率加权 LRU（频率越低越先淘汰；llama.cpp 2026-03 的 LFRU）
  pinned           频率常驻（top-P 永不淘汰；借鉴 colibri 的 learned pinned hot-store）
  pinned_prefetch  pinned + 域画像预取（借鉴 MoE-Infinity 的激活感知预取）

⚠ 诚实边界：策略收益随「缓存 ≪ 工作集」放大；稳态下 LRU 已经很好。
实测（benchmarks/policy_bench.py）：连续单域几乎无差别，切域场景尖峰明显变稳。
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

MODES = ("lru", "slru", "prefetch", "pinned", "pinned_prefetch")


@dataclass
class PolicyConfig:
    capacity: int = 64          # 快层槽位数（GPU 专家槽）
    mode: str = "lru"           # 见 MODES
    pin_frac: float = 0.4       # pinned 模式下 capacity 里用于 pin 的比例
    prefetch_q: int = 16        # 切域时预取多少个专家
    freq_ema: float = 0.99      # 频率 EMA（≈ 长期路由占比，与 router.frac_ema 同源）

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"未知 mode={self.mode!r}，可选 {MODES}")
        if self.capacity < 1:
            raise ValueError("capacity 必须 >= 1")


@dataclass
class PolicyStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    prefetched: int = 0
    prefetch_used: int = 0

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0

    def as_dict(self) -> Dict[str, float]:
        return dict(hits=self.hits, misses=self.misses, evictions=self.evictions,
                    prefetched=self.prefetched, prefetch_used=self.prefetch_used,
                    hit_rate=round(self.hit_rate, 4))


class ResidencyPolicy:
    """固定容量的快层缓存 + 可选 pin / 预取。纯 CPU、零依赖。"""

    def __init__(self, n_experts: int, cfg: PolicyConfig) -> None:
        self.n = int(n_experts)
        self.cfg = cfg
        self.cache: "OrderedDict[int, None]" = OrderedDict()
        self.freq = np.zeros(self.n, dtype=np.float64)
        self.pin: Set[int] = set()
        self.stats = PolicyStats()
        self._prefetched: Set[int] = set()

        self._lru_cap = int(cfg.capacity)
        if cfg.mode in ("pinned", "pinned_prefetch"):
            n_pin = max(1, int(round(cfg.capacity * cfg.pin_frac)))
            self._lru_cap = max(1, cfg.capacity - n_pin)

    # ------------------------------------------------------------------ 访问
    def access(self, eid: int) -> bool:
        eid = int(eid)
        self.freq *= self.cfg.freq_ema
        self.freq[eid] += 1.0

        if eid in self.pin:
            self.stats.hits += 1
            self._consume_prefetch(eid)
            return True

        if eid in self.cache:
            self.cache.move_to_end(eid)
            self.stats.hits += 1
            self._consume_prefetch(eid)
            return True

        self.stats.misses += 1
        self._insert(eid)
        return False

    def _consume_prefetch(self, eid: int) -> None:
        if eid in self._prefetched:
            self._prefetched.discard(eid)
            self.stats.prefetch_used += 1

    def _insert(self, eid: int) -> None:
        self.cache[eid] = None
        self.cache.move_to_end(eid)
        if len(self.cache) > self._lru_cap:
            self.cache.popitem(last=False)
            self.stats.evictions += 1

    # ------------------------------------------------------------------ pin
    def refresh_pin(self, k: Optional[int] = None) -> List[int]:
        """按频率 EMA 重选热点并 pin 住（借鉴 colibri 的 learned hot-store）。

        返回**新增**被 pin 的专家。建议 warmup 后每 N 个 token 调一次。
        """
        if self.cfg.mode not in ("pinned", "pinned_prefetch"):
            return []
        k = k if k is not None else max(1, int(round(self.cfg.capacity * self.cfg.pin_frac)))
        k = min(k, self.n)
        new_pin = set(int(i) for i in np.argsort(-self.freq)[:k])
        added = sorted(new_pin - self.pin)
        self.pin = new_pin
        for eid in list(self.cache):          # pin 住的不再占 LRU 槽
            if eid in self.pin:
                del self.cache[eid]
        return added

    # ------------------------------------------------------------------ 预取
    def switch_domain(self, hot_ids: Sequence[int]) -> List[int]:
        """切域时预取该域热点（在路由之前调用）。

        真实引擎里应发起**异步**加载并与当前计算重叠；本实现同步返回。
        """
        if self.cfg.mode not in ("prefetch", "pinned_prefetch"):
            return []
        candidates = [int(e) for e in list(hot_ids)[: self.cfg.prefetch_q]
                      if int(e) not in self.pin and int(e) not in self.cache]
        for eid in candidates:
            # 正常插入（必要时按 LRU 轮换掉冷项）——预取的意义就是把冷项换成热项
            self._insert(eid)
        # Reconcile once after the loop: only report experts that are still resident.
        # Deciding per-insertion also reports experts that LRU immediately evicts again,
        # which makes the host move bytes that retain() will drop — one wasted I/O.
        #   （插入过程中先到的会被后到的挤掉；若逐个判断就会把"随后被淘汰"的专家
        #     也报出去，宿主真搬上快层 ⇒ 白搬一次 I/O。本条由 tests/test_plugin.py 发现）
        resident = self.resident()
        out = [e for e in candidates if e in resident]
        self._prefetched.update(out)
        self.stats.prefetched += len(out)
        return out

    # ------------------------------------------------------------------ 观测
    def resident(self) -> Set[int]:
        return set(self.cache) | set(self.pin)

    def top_hot(self, k: int = 16) -> List[int]:
        return [int(i) for i in np.argsort(-self.freq)[:k]]


class SLRUPolicy:
    """分段 LRU（Segmented LRU）：probation + protected 两段。

    对齐 llama.cpp #20757 的两层缓存提案。相比纯 LRU，它把"只命中过一次"的新条目
    留在 probation（优先被淘汰），把"命中过两次以上"的放进 protected（更难被挤掉）。
    好处：突发的新专家不会把稳定的热专家挤走。
    """

    def __init__(self, n_experts: int, cfg: PolicyConfig) -> None:
        self.n = int(n_experts)
        self.cfg = cfg
        prot_cap = max(1, int(round(cfg.capacity * 0.8)))
        self.prot_cap = prot_cap
        self.prob_cap = max(1, cfg.capacity - prot_cap)
        self.protected: "OrderedDict[int, None]" = OrderedDict()
        self.probation: "OrderedDict[int, None]" = OrderedDict()
        self.freq = np.zeros(self.n, dtype=np.float64)
        self.pin: Set[int] = set()
        self.stats = PolicyStats()

    # ------------------------------------------------------------------ 访问
    def access(self, eid: int) -> bool:
        eid = int(eid)
        self.freq *= self.cfg.freq_ema
        self.freq[eid] += 1.0

        if eid in self.pin:
            self.stats.hits += 1
            return True
        if eid in self.protected:
            self.protected.move_to_end(eid)
            self.stats.hits += 1
            return True
        if eid in self.probation:
            del self.probation[eid]
            self._promote(eid)
            self.stats.hits += 1
            return True

        self.stats.misses += 1
        self.probation[eid] = None
        self._trim()
        return False

    def _promote(self, eid: int) -> None:
        self.protected[eid] = None
        self.protected.move_to_end(eid)
        while len(self.protected) > self.prot_cap:
            victim, _ = self.protected.popitem(last=False)
            self.probation[victim] = None
            self.probation.move_to_end(victim)
        self._trim()

    def _trim(self) -> None:
        while len(self.probation) > self.prob_cap:
            self.probation.popitem(last=False)
            self.stats.evictions += 1

    # ------------------------------------------------------------------ 兼容接口
    def refresh_pin(self, k: Optional[int] = None) -> List[int]:
        return []                      # SLRU 不 pin（由 protected 段隐式承担）

    def switch_domain(self, hot_ids: Sequence[int]) -> List[int]:
        return []                      # SLRU 不预取

    def resident(self) -> Set[int]:
        return set(self.protected) | set(self.probation) | set(self.pin)

    def top_hot(self, k: int = 16) -> List[int]:
        return [int(i) for i in np.argsort(-self.freq)[:k]]


def make_policy(n_experts: int, cfg: PolicyConfig):
    """按 cfg.mode 造出策略对象（对外统一入口）。"""
    if cfg.mode == "slru":
        return SLRUPolicy(n_experts, cfg)
    return ResidencyPolicy(n_experts, cfg)
