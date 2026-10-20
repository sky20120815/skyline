"""llama.cpp 接入（MIT）。

llama.cpp 的 MoE 卸载是**静态**的（`--n-cpu-moe N` 按层切），今天还没有"按使用热度 pin"的机制
（社区在 issue #20757 / PR #698 讨论两层缓存与频率加权 LRU）。

因此本适配器提供两件**现在就能用**的东西：
1. **热集清单导出**：把本插件学到的域画像导出成一份可读的清单，供 patch 或人工决策使用
2. **命令行配方**：给出在当前 llama.cpp 上跑分层 MoE 的可用开关组合

并诚实说明：真正的"pin 到 GPU 缓冲"需要改 llama.cpp 的专家放置路径（插入点见 hook_points）。
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional


def export_hotset(profile, path: str, top: int = 64) -> str:
    """把域画像导出成热集清单（**本仓库自己的格式**，不是 colibri/llama.cpp 的内部格式）。"""
    payload = {
        "format": "moe-residency/hotset/1",
        "note": "expert ids per domain; 用法见 README「作为插件使用」",
        "domains": {d: profile.hint(d, q=top) for d in profile.domains()},
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def cli_recipe(n_gpu_moe: int = 0, n_cpu_moe: int = 0, cache_type_k: str = "q8_0",
               cache_type_v: str = "q8_0", threads: Optional[int] = None) -> List[str]:
    """当前 llama.cpp 上可用的分层 MoE 命令行组合。"""
    cmd = ["llama-cli", "-m", "<model>.gguf", "--n-gpu-layers", "99",
           "--n-cpu-moe", str(n_cpu_moe), "--n-gpu-moe", str(n_gpu_moe),
           "--cache-type-k", cache_type_k, "--cache-type-v", cache_type_v]
    if threads:
        cmd += ["-t", str(threads)]
    return cmd


def hook_points() -> List[str]:
    """要实现"热度 pin"需要改的位置（概念级指引）。"""
    return [
        "专家放置逻辑（`--n-cpu-moe` 对应的 MoE 层 offload 决策处）⇒ 改为按热集清单决定 GPU/CPU",
        "路由统计：把每 token 选中的 expert id 累加进一个 per-layer 计数表（我们的 profile 可复用）",
        "淘汰：把静态切分换成 LRU/LFRU（我们的 policy.py 是可直接照搬的参考实现）",
    ]


def notes() -> List[str]:
    return [
        "上游 llama.cpp 已有 ik_llama.cpp PR #698（只 offload 被激活的专家）与社区的 LFRU 讨论，"
        "接入前请先确认当前版本的专家放置实现。",
        "本仓库不提供针对 llama.cpp 的可直接 apply 的补丁（避免给出与上游版本不匹配的 diff）。",
    ]
