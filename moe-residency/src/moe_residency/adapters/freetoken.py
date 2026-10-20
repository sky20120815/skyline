"""FreeToken 接入（UC Berkeley/MIT，Apache-2.0）。

FreeToken 的强项是 **q\\* 带宽自适应**（在 PCIe 搬运与 CPU 直算之间实时分流）、
**全层双缓冲**、**弹性显存**与**语义锚点**（agent 任务边界处的 KV 复用）。

**我们能补的增量**：它的 semantic anchor 已经知道"任务边界在哪"，
但那里**只缓存 KV、不预取专家**。本插件正好可以在同一处切域画像并预取该域热专家。

⚠ FreeToken 的专家缓存为 global LRU（记"最近"不记"最常"），本插件的 pinned 策略
是"记最常"，两者正交、可叠加。
"""
from __future__ import annotations

from typing import List


def hook_points() -> List[str]:
    return [
        "semantic anchor / 任务边界事件 ⇒ 调用 before_route(domain) 切域画像 + 预取该域热集",
        "每 token 路由后 ⇒ on_route(expert_ids)；未命中的专家交给 FreeToken 自己的 q* 分流",
        "每 token 结束后 ⇒ after_token()",
    ]


def integration_notes() -> List[str]:
    return [
        "不要改 q*：它决定 miss 走 PCIe 还是 CPU，与本插件正交（我们只改变 miss 集合的大小）。",
        "FreeToken 的热路径是 Rust/CUDA，Python 插件只能挂在其上层事件上；"
        "要做到 per-layer 粒度需要上游改动。",
        "若上游愿意接受：把『按域热集』做成 q* 的一个输入（像 q* 接受实测带宽一样接受实测热度）是最干净的接法。",
    ]
