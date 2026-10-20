"""插件生命周期：让宿主引擎用 **3 个调用点**接上策略与验证。

设计目标：宿主引擎（colibri / llama.cpp / FreeToken / 自研）不需要了解
"pin / LRU / 画像 / 逐位验证"这些概念，只要在三个时机回调即可：

    ① 切任务/切域时（**路由之前**）   before_route(domain)
    ② 每 token 路由出专家后          on_route(expert_ids)
    ③ 每 token 计算完之后            after_token()

插件通过标准 Python entry point 发现（见 pyproject 的
`[project.entry-points."moe_residency.plugins"]`），宿主无需 import 本包的路径：

    from importlib.metadata import entry_points
    ep = next(e for e in entry_points(group="moe_residency.plugins"))
    plugin = ep.load()()
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

from .policy import PolicyConfig, make_policy
from .profile import DomainProfile
from .verify import BitExactVerifier


class ResidencyPlugin:
    """专家驻留插件：策略 + 域画像 + 可选的逐位验证。

    典型用法（宿主侧）::

        plugin = ResidencyPlugin(n_experts=19456, capacity=64, mode="prefetch")
        ...
        for token in stream:
            plugin.before_route(domain=detect_task(token))   # ①
            ids = router.topk(token)                        # 宿主自己的路由
            misses = plugin.on_route(ids)                   # ② 返回需要搬的专家
            for eid in misses:
                block = engine.load_expert(eid)            # 宿主负责搬运
            out = engine.compute(token)                    # 宿主自己的计算
            plugin.after_token()                           # ③
    """

    name = "moe-residency"
    version = "0.1.0"

    def __init__(self, n_experts: int, capacity: int = 64, mode: str = "prefetch",
                 pin_frac: float = 0.3, prefetch_q: int = 12,
                 profile_path: Optional[str] = None,
                 verifier: Optional[BitExactVerifier] = None,
                 run_name: str = "run") -> None:
        self.n_experts = int(n_experts)
        self.cfg = PolicyConfig(capacity=int(capacity), mode=mode,
                                 pin_frac=float(pin_frac), prefetch_q=int(prefetch_q))
        self.policy = make_policy(self.n_experts, self.cfg)
        self.profile = (DomainProfile.load(profile_path) if profile_path
                        and os.path.exists(profile_path) else DomainProfile(self.n_experts))
        self.profile_path = profile_path
        self.verifier = verifier
        self.run_name = run_name
        self.current_domain: Optional[str] = None
        self._last_route: List[int] = []

    # ------------------------------------------------------------------ 生命周期
    def before_route(self, domain: Optional[str] = None) -> List[int]:
        """① 切任务/切域时调用（**必须早于路由**）。返回建议预取的专家 id。"""
        if domain is not None and domain != self.current_domain:
            self.current_domain = domain
            hot = self.profile.hint(domain, q=self.cfg.prefetch_q)
            prefetched = self.policy.switch_domain(hot)
            if prefetched:
                self._note("prefetch", len(prefetched), domain)
            return list(prefetched)
        return []

    def on_route(self, expert_ids: Sequence[int]) -> Dict[int, bool]:
        """② 路由出专家后调用。返回 {eid: 是否 miss}（True = 需要宿主去搬）。"""
        misses: Dict[int, bool] = {}
        for eid in expert_ids:
            hit = self.policy.access(int(eid))
            misses[int(eid)] = not hit
        self._last_route = [int(e) for e in expert_ids]
        if self.current_domain is not None:
            self.profile.note(self.current_domain, self._last_route)
        return misses

    def after_token(self) -> None:
        """③ 每 token 结束后调用：刷新热点 / 周期保存画像。"""
        if self.policy.stats.hits + self.policy.stats.misses >= 0 and \
                (self.policy.stats.hits + self.policy.stats.misses) % 200 == 0:
            self.policy.refresh_pin()
        if self.profile_path and (self.policy.stats.hits + self.policy.stats.misses) % 2000 == 0:
            try:
                self.profile.save(self.profile_path)
            except Exception:                                  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ 验证
    def record(self, logits) -> str:
        """记录一次前向的 logits 摘要（供逐位对拍）。"""
        if self.verifier is None:
            raise RuntimeError("未启用 verifier")
        return self.verifier.record(self.run_name, logits)

    def assert_bitexact(self, other_run: str) -> None:
        if self.verifier is None:
            raise RuntimeError("未启用 verifier")
        self.verifier.assert_bitexact(self.run_name, other_run)

    # ------------------------------------------------------------------ 观测
    def stats(self) -> Dict[str, Any]:
        return {
            "plugin": f"{self.name}@{self.version}",
            "mode": self.cfg.mode,
            "capacity": self.cfg.capacity,
            "policy": self.policy.stats.as_dict(),
            "domains": len(self.profile.domains()),
            "resident": len(self.policy.resident()),
        }

    def _note(self, *a) -> None:
        if os.environ.get("MOE_RESIDENCY_VERBOSE"):
            print("[moe-residency]", *a)

    # ------------------------------------------------------------------ 清单
    @staticmethod
    def manifest() -> Dict[str, Any]:
        """插件清单（供宿主/仓库索引使用）。"""
        return {
            "name": "moe-residency",
            "version": ResidencyPlugin.version,
            "entry_point": "moe_residency.plugin:ResidencyPlugin",
            "capabilities": ["expert-residency-policy", "domain-profile-prefetch",
                              "bit-exact-verification"],
            "modes": ["lru", "slru", "prefetch", "pinned", "pinned_prefetch"],
            "license": "MIT",
        }


def load_plugin(**kwargs) -> ResidencyPlugin:
    """entry point 的加载函数（宿主用 `entry_points(...).load()()` 调用）。"""
    return ResidencyPlugin(**kwargs)


def _manifest_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "moe_residency.plugin.json")


def write_manifest(path: Optional[str] = None) -> str:
    """把插件清单写到仓库根（CI 可校验它与代码一致）。"""
    p = path or _manifest_path()
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(ResidencyPlugin.manifest(), fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    return p
