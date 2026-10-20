"""moe_residency — 专家驻留策略 + 逐位不变验证（引擎无关的参考实现）。

这个包**不是**一个推理引擎，它只做两件事：
  1. 决定「哪些专家该留在快层、下一个该预取谁」（policy）
  2. 证明「分层放置只改速度、不改语义」（verify）

配套的 `engine.py` 是一个 ~200 行的**玩具引擎**，只为把上面两件事跑通、可验证，
不追求跑真实大模型。真实部署请接入 colibri / FreeToken / llama.cpp（见 docs/INTEGRATION.md）。
"""

from .policy import PolicyConfig, ResidencyPolicy, SLRUPolicy, make_policy
from .profile import DomainProfile
from .verify import BitExactVerifier, tensor_digest
from .plugin import ResidencyPlugin, load_plugin

__all__ = [
    "PolicyConfig",
    "ResidencyPolicy",
    "SLRUPolicy",
    "make_policy",
    "DomainProfile",
    "BitExactVerifier",
    "tensor_digest",
    "ResidencyPlugin",
    "load_plugin",
]
__version__ = "0.1.0"
