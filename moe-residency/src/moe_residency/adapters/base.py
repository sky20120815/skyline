"""宿主引擎需要实现的最小接口（两个方法 + 一个属性）。

任何 MoE 推理引擎只要能把"取专家"和"切任务"暴露出来，就能接上本仓库的策略与验证。
具体引擎的接入说明见 docs/INTEGRATION.md。
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class ExpertSource(Protocol):
    """慢层专家来源（RAM 或磁盘）。"""

    def fetch(self, eid: int) -> dict:
        """取出一个专家块的权重（应保持原始 dtype，避免引入数值偏差）。"""
        ...


@runtime_checkable
class ResidencyHost(Protocol):
    """宿主引擎需要暴露的三件事。"""

    n_experts: int

    def route(self, token_input: np.ndarray) -> Sequence[int]:
        """返回该 token 选中的专家 id（top_k）。"""
        ...

    def load_expert(self, eid: int) -> None:
        """把专家搬进快层（同步或异步均可，异步时需保证在本 token 计算前完成）。"""
        ...

    def release_expert(self, eid: int) -> None:
        """把专家从快层释放（若引擎自己管理淘汰，可不实现）。"""
        ...
