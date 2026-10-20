"""逐位不变验证：证明「分层放置只改速度、不改语义」。

为什么需要它：所有 offload 引擎都声称"放置不影响输出"，但**很少有可复用的验证工具**。
本模块给你一个最小但严格的做法：
  1. 记录每次前向的 logits 摘要（sha256 of float32 bytes）
  2. 对比「全驻留」与「分层」两条路径的摘要序列
  3. 必须**逐位相同**，不是"误差很小"

这与 colibri 的 token-exact 验证、本项目的 `test_paging_invariance.py` 是同一个判据，
区别是这里做成了与引擎无关、可被 CI 直接调用的库。
"""
from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def tensor_digest(t: np.ndarray) -> str:
    """张量的逐位摘要（强制 float32 little-endian，避免 dtype/端序差异造成假不同）。"""
    a = np.ascontiguousarray(np.asarray(t, dtype=np.float32))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


class BitExactVerifier:
    """记录并比对多次运行的结果摘要。"""

    def __init__(self) -> None:
        self.runs: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------ 记录
    def start(self, name: str) -> None:
        self.runs[name] = []

    def record(self, name: str, logits: np.ndarray) -> str:
        d = tensor_digest(logits)
        self.runs.setdefault(name, []).append(d)
        return d

    # ------------------------------------------------------------------ 比对
    def compare(self, a: str, b: str) -> Tuple[bool, Dict[str, object]]:
        """返回 (是否逐位相同, 差异详情)。"""
        A, B = self.runs.get(a), self.runs.get(b)
        if A is None or B is None:
            raise KeyError(f"缺少运行记录：{a if A is None else b}")
        if len(A) != len(B):
            return False, {"reason": "step_count_mismatch", "a": len(A), "b": len(B)}
        bad = [i for i, (x, y) in enumerate(zip(A, B)) if x != y]
        info = {
            "steps": len(A),
            "mismatched_steps": len(bad),
            "first_mismatch": bad[0] if bad else None,
            "digest_a": A[:4],
            "digest_b": B[:4],
        }
        return (not bad), info

    def assert_bitexact(self, a: str, b: str) -> None:
        ok, info = self.compare(a, b)
        if not ok:
            raise AssertionError(f"非逐位一致：{info}")


def digest_sequence(arrs: Iterable[np.ndarray]) -> List[str]:
    return [tensor_digest(x) for x in arrs]
