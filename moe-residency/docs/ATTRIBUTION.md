# 借鉴与致谢

本仓库**未复制任何第三方代码**：所有实现均从零重写（numpy + 标准库）。
以下项目提供了机制、设计约定或工程先例。若你在使用中发现问题，责任在本仓库，不在这项目。

| 项目 | 仓库 | 许可 | 本仓库借鉴了什么 | 本仓库**没有**做什么 |
|---|---|---|---|---|
| **colibri** | `JustVugg/colibri` | Apache-2.0 | ① 三级分层（VRAM 热 / RAM 温 / NVMe 冷）；② **learned pinned hot-store**（按使用热度 pin 住专家，持久化 `.coli_usage`）；③ 一层超前预取；④ **「placement 只改速度、不改语义」的硬承诺**，以及 token-exact 对拍验证的做法 | 未实现 MTP 投机解码、MLA KV 压缩、双 SSD 条带、O_DIRECT、纯 C 内核 |
| **FreeToken** | `FlashML-org/FreeToken` | Apache-2.0 | ① GPU 作为**弹性专家缓存** + 主机内存放完整专家池；② **q\* 带宽自适应**的分流思想；③ 全层双缓冲；④ 语义锚点（本仓库未实现，仅在设计上参考）| 未实现 q\* 的闭式求解、双缓冲、FTW 权重格式 |
| **Strata** | `Niko1221/Strata` | MIT | ① GPU 放稠密部分 + 热专家；② **CPU 就地计算冷专家**的分工；③ 全池驻留 RAM | 未实现 CPU 向量化内核、投机解码、n-gram 表 |
| **KTransformers** | `kvcache-ai/ktransformers` | Apache-2.0 | ① CPU/GPU 异构协同（注意力在 GPU、专家在 CPU）；② 静态布局 vs 动态调度的取舍讨论 | 未实现 AMX/AVX-512 内核、SGLang 后端 |
| **PowerInfer** | `SJTU-IPADS/PowerInfer` | — | **热/冷分离**的粒度思想（它做到**神经元**级，本仓库只到**专家**级）| 未实现神经元级预测器 |
| **MoE-Infinity** | `EfficientMoE/MoE-Infinity` | — | **激活感知的专家预取** —— `profile.py` 的直接思想来源（按域统计路由频次，路由前预取）| 未实现其离线 profiler 与 trace 回放 |
| **llama.cpp** | `ggml-org/llama.cpp` | MIT | ① 两层专家缓存提案（issue #20757）的 **SLRU 分段**思路；② `--n-cpu-moe` 的卸载形态 | 未修改 llama.cpp 源码（接入方案见 `docs/INTEGRATION.md`）|
| **vLLM** | `vllm-project/vllm` | Apache-2.0 | PR #37190 中「GPU 侧专家缓存 + 频次淘汰」的思路 | 未实现 PagedAttention / continuous batching |
| **Bonsai 2 / BitNet** | — | — | 三值量化的容量换算（1.58–4.125 bit/参数）| **未实现**任何量化（本仓库只验证"放置不改语义"，不验证"压缩不改语义"）|

## 关键思想的出处（逐条）

| 本仓库的机制 | 思想出处 |
|---|---|
| 三级存储（fast / warm / disk） | colibri（最完整）、Strata（SSD 只放表）、FreeToken（两级 + PCIe）|
| 频率常驻（`pinned`） | colibri 的 learned pinned hot-store（把使用热度持久化到 `.coli_usage`）|
| 域画像预取（`prefetch`） | MoE-Infinity 的激活感知预取；本仓库改成"按域条件化"（agent 切任务时更准）|
| SLRU 分段缓存（`slru`） | llama.cpp issue #20757 的提案；vLLM 的 LFRU 变体 |
| 逐位不变验证（`verify.py`） | colibri 的 token-exact 对拍；本项目 `test_paging_invariance.py` 的"分页不影响语义"不变量 |
| "停顿 miss" vs "预取 I/O" 两类指标 | 本仓库提出（用于避免用 I/O 总量错误评价预取）|

## 许可说明

- 本仓库：**MIT**。
- 上述项目多为 Apache-2.0 / MIT，与本仓库兼容。**由于没有复制代码，不触发其署名/传染条款**；
  上表与 README 的致谢是出于学术规范与尊重，而非法律要求。
- 若未来接入上述任一项目的源码（例如给 llama.cpp 提 PR），该部分代码将遵循其原许可。

## 如何引用

如果这个验证器或策略实现对你的工作有帮助，请优先引用上表中的原始项目；
本仓库更适合作为"验证工具"被提及：

```bibtex
@misc{moe_residency_2026,
  title  = {moe-residency: bit-exact expert residency policies for MoE offloading},
  year   = {2026},
  note   = {Reference implementation and verification harness}
}
```
