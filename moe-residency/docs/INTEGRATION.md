# 接入现有引擎

本仓库只做策略与验证，**接入方式是在宿主引擎里插入两个调用点 + 一次对齐**：

```python
# 1) 切域/换任务时（路由之前）
policy.switch_domain(profile.hint(domain, q=prefetch_q))   # 返回需预取的 eid
#    → 宿主引擎应在此**异步**发起加载，并与接下来的计算重叠

# 2) 每个专家被路由到时
hit = policy.access(eid)
if not hit:
    block = store.fetch(eid)        # 从 RAM 或磁盘取
    store.put(eid, block)

# 3) 每个 token 结束时对齐驻留集合（唯一真相在 policy）
store.retain(policy.resident())

# 4) 周期性刷新热点（warmup 之后）
policy.refresh_pin()
```

可选但强烈建议：每次前向用 `BitExactVerifier` 记录 logits 摘要，
开一个「全驻留」对照进程，逐位比对（CI 里跑）。

---

## colibri（`JustVugg/colibri`，纯 C，Apache-2.0）

- **插入位置**：它的 per-layer LRU 专家缓存管理器。
- **pin**：colibri 已有 `.coli_usage` 的 learned hot-store，本仓库的 `pinned` 只是同一机制的
  Python 参考；若要贡献，价值在**按域条件化**（它的热度是全局的）。
- **prefetch**：它的预取是「一层超前」（layer-ahead）；本仓库的 `prefetch` 是「域切换时」，
  两者互补 —— 前者解决层内时间局部性，后者解决任务切换的一次性冷启动。
- **验证**：它已有 token-exact 对拍；可用 `verify.py` 的思路做独立第三方校验。

## llama.cpp（`ggml-org/llama.cpp`，MIT）

- **插入位置**：MoE 专家卸载调度（`--n-cpu-moe` 的放置逻辑）+ `llama_decode` 的层循环。
- **建议补丁形态**：新增 `--hot-expert-pin N`（按使用计数 pin 前 N 个到 GPU 缓冲）、
  `--expert-usage-file path`（持久化计数，冷启动免 warmup）。
- **SLRU**：issue #20757 已提出两层缓存；本仓库 `slru` 可直接作为参考实现。
- **验证**：同一 prompt、同一 seed，比较 `--n-cpu-moe` 不同取值下的 logits 是否逐位一致。

## FreeToken（`FlashML-org/FreeToken`，Apache-2.0）

- **插入位置**：global LRU 专家缓存 + semantic anchor 钩子。
- **互补点**：它的 semantic anchor 缓存**KV**，但不预取**专家**；本仓库的
  `prefetch` 正好补这个缺口（agent 切换任务时既有 KV 复用、又有专家预热）。
- **不要动 q\***：它的带宽自适应分流是核心，我们的策略只改变 miss 集合的大小，
  两者正交。

## KTransformers（`kvcache-ai/ktransformers`，Apache-2.0）

- 它的 CPU 就地计算冷专家，本身就是"慢层即计算层"；
  本仓库的 `pinned`/`prefetch` 可减少 CPU 侧被迫计算的次数（把热专家提前搬上 GPU）。
