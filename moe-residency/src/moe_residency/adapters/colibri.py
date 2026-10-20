"""colibri 接入（Apache-2.0，纯 C）。

**重要**：colibri 的 `.coli_usage` 文件格式**未在其文档中公开**，
因此本适配器**不猜测该格式**，只做两件确定的事：

1. 给出**只用已公开开关**的环境变量配方（这些开关见 colibri 的 docs/ENVIRONMENT.md）
2. 指出我们的「域条件化热集」应该挂在 colibri 的哪个事件上（任务边界 / semantic anchor）

⚠ colibri 已有 `PILOT`（router-piloted 预取）与 `PILOT_EVICT_GUARD`
（防止预取的专家在被用之前被淘汰）—— 后者正是"预取必须同步进驻留集合"这条纪律。
我们能补的增量是**按域条件化**：切任务时热集不同，而 colibri 的热度是全局的。
"""
from __future__ import annotations

from typing import Dict, List, Optional


def env_recipe(model_dir: str, pin_gb: float = 16.0, prefetch: int = 2,
               pipe: bool = True, pipe_workers: int = 8, usage_decay: Optional[float] = None,
               usage_save: bool = True, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """生成一份 colibri 的环境变量配方（全部为 colibri 文档公开的开关）。"""
    env: Dict[str, str] = {
        "COLI_MODEL": model_dir,
        "COLI_USAGE": f"{model_dir}/.coli_usage",
        "PIN": "auto",                 # 从 .coli_usage / stats.txt 播种热集
        "PIN_GB": str(pin_gb),          # 常驻热集预算
        "PREFETCH": str(prefetch),      # 流式专家的预取深度
        "PILOT": "1",                   # 路由引导的跨层预取
        "PILOT_REAL": "1",              # 保持数值语义的预取（值保持 ⇒ 逐位不变）
        "PILOT_EVICT_GUARD": "1",       # 预取的专家在被用前不被淘汰
        "USAGE_SAVE": "1" if usage_save else "0",
    }
    if pipe:
        env["PIPE"] = "1"
        env["PIPE_WORKERS"] = str(pipe_workers)
    if usage_decay is not None:
        env["COLI_USAGE_DECAY"] = str(usage_decay)
    if extra:
        env.update(extra)
    return env


def decay_for_half_life(half_life_runs: float) -> float:
    """把"半衰期（按运行次数）"换算成 colibri 的 `COLI_USAGE_DECAY`（每次运行的乘数）。

    colibri 文档：COLI_USAGE_DECAY 是"每次运行给全部历史 counts 乘一个系数"，
    默认 1.0（不衰减）⇒ 不设它，排名会冻结（约 1800 万次选择后一次新选择只动 0.2%）。
    """
    if half_life_runs <= 0:
        return 1.0
    return float(0.5 ** (1.0 / half_life_runs))


def hook_points() -> List[str]:
    """本插件可挂载的事件点（对应 colibri 的概念）。"""
    return [
        "agent 任务边界 / semantic anchor 处 ⇒ 调用 before_route(domain) 切换域画像并预取该域热集",
        "每 token 路由出专家后 ⇒ on_route(expert_ids) 统计命中/未命中",
        "每 token 结束后 ⇒ after_token() 刷新热点",
    ]


def notes() -> List[str]:
    return [
        "colibri 已有 PILOT 预取与 PILOT_EVICT_GUARD，我们的增量是『按域条件化』。",
        "`.coli_usage` 的文件格式未公开 ⇒ 不要试图直接写它；用 PIN=auto + STATS 让 colibri 自己维护。",
        "PILOT_REAL=1 才能保证预取不改变数值语义（与我们的逐位不变目标一致）。",
    ]
