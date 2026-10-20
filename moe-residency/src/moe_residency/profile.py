"""域→专家 画像（哪个域偏爱哪些专家），EMA 平滑 + JSON 持久化。

借鉴 MoE-Infinity 的「激活感知预取」思路：先知道"这类输入历史上路由到谁"，
再在路由**发生之前**把它们预取进快层。与 colibri 的 learned hot-store 的区别是：
colibri 学的是**全局**热度，这里学的是**按域条件化**的热度（agent 切任务时更准）。
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence


class DomainProfile:
    """`domain -> {expert_id: 频次}`，带 EMA 平滑与磁盘持久化。"""

    def __init__(self, n_experts: int, ema: float = 0.95) -> None:
        self.n = int(n_experts)
        self.ema = float(ema)
        self.table: Dict[str, Dict[int, float]] = {}

    # ------------------------------------------------------------------ 更新
    def note(self, domain: str, eids: Sequence[int], ema: Optional[float] = None) -> None:
        """记录一次路由结果（每个 token 调一次即可）。"""
        if domain is None:
            return
        a = self.ema if ema is None else float(ema)
        prof = self.table.setdefault(str(domain), {})
        for k in list(prof):                      # 旧热度按 a 衰减
            prof[k] *= a
            if prof[k] < 1e-6:
                del prof[k]
        for e in eids:
            e = int(e)
            prof[e] = prof.get(e, 0.0) + 1.0

    # ------------------------------------------------------------------ 查询
    def hint(self, domain: str, q: int = 16) -> List[int]:
        """该域历史上最热的 q 个专家（供预取）。未知域返回空。"""
        prof = self.table.get(str(domain))
        if not prof:
            return []
        return [int(e) for e, _ in sorted(prof.items(), key=lambda kv: -kv[1])[:q]]

    def domains(self) -> List[str]:
        return sorted(self.table)

    # ------------------------------------------------------------------ 持久化
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = {
            "n_experts": self.n,
            "ema": self.ema,
            "table": {d: {str(k): round(v, 6) for k, v in p.items()}
                      for d, p in self.table.items()},
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "DomainProfile":
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        obj = cls(int(raw["n_experts"]), float(raw.get("ema", 0.95)))
        obj.table = {d: {int(k): float(v) for k, v in p.items()}
                     for d, p in raw.get("table", {}).items()}
        return obj

    def __repr__(self) -> str:
        return f"DomainProfile(n={self.n}, domains={len(self.table)})"
