# 变更日志

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)；
本项目尚未发布稳定版本，版本号即 `pyproject.toml` 中的 `version`。

## [0.1.0] — 2026-10-20

首个可发布版本。

### 新增
- **逐位不变验证器** `BitExactVerifier`：对每次前向的 logits 做 float32 sha256，
  逐位比对「全驻留」与「分层」两条路径（仓库的核心主张，由 `tests/test_engine.py` 守护）
- **5 种专家驻留策略**：`lru` / `slru`（分段）/ `prefetch`（域画像预取）/
  `pinned`（频率常驻）/ `pinned_prefetch`
- **域画像** `DomainProfile`：域→专家频次 EMA + JSON 持久化
- **插件形态**：注册在 entry point 组 `moe_residency.plugins`；
  宿主只需 3 个调用点（`before_route` / `on_route` / `after_token`）
- **参考引擎** `engine.py`：三级存储（fast / warm / 真实磁盘）+ 单块 MoE 确定性前向
- **引擎适配器**：colibri（环境变量配方）、llama.cpp（热集导出 + 命令行配方）、
  FreeToken（semantic anchor 挂载点）
- **硬件矩阵** `docs/HARDWARE_MATRIX.md` + `benchmarks/theoretical_speed.py`：
  各档位「最大可跑模型」与理论速度，并附实测对照与校准说明
- **测试套件** 53 个用例（标准库 unittest，零额外依赖）

### 修复
- `switch_domain` 曾把「随后又被 LRU 淘汰」的专家也报给宿主，
  会导致一次白搬的 I/O。改为**循环结束后统一结算**，只返回确实驻留的专家。
  （由 `tests/test_plugin.py` 的小缓存用例发现）

### 已知限制
- 不含任何量化；只验证「放置不改语义」，不验证「压缩不改语义」
- 玩具引擎的性能数字不具参考价值
- 策略收益**强负载相关**：实测 `prefetch` 在切域负载 −24% 停顿 miss，
  但在合成长尾负载 ≈0；`pin`/`slru` 在部分负载上反而更差
